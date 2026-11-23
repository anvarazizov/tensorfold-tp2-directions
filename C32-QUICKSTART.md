# Reproduce a mixed batch of 32 on TensorFold TP2

Download the [C32 build](https://github.com/anvarazizov/tensorfold-tp2-directions/releases/download/v1.0.0/tf-runtime-directions-tp2-gb10-c32.tar.gz) and its [SHA-256 sidecar](https://github.com/anvarazizov/tensorfold-tp2-directions/releases/download/v1.0.0/tf-runtime-directions-tp2-gb10-c32.tar.gz.sha256). The release archive includes both Linux aarch64 executables, the sm_121 kernel, the AOT kit, directions and the qualification script. It uses the original ×1 kernel for the concurrency test. The ×2.5 behavioral experiment in the field report used a separate kernel.

## Prepare two idle GB10 hosts

Use two 128 GB GB10s with working RoCEv2 GID 3, NVIDIA Container Toolkit and a compatible arm64 CUDA 13 image on both. Supply the original EXL3 2.5 bpw model view, matching per-rank caches, token map and native Engram shards. The archive README identifies the pinned model revision and directory layout. Both the engine and the worker must come from this release.

On the operator machine, from the download directory, run in a fresh working directory containing the two release files:

```sh
shasum -a 256 -c tf-runtime-directions-tp2-gb10-c32.tar.gz.sha256
tar -xzf tf-runtime-directions-tp2-gb10-c32.tar.gz
cd tf-runtime-directions-tp2-gb10-c32
python3 portable.py verify
cp target.example.json ../target-tp2.json
```

Set both hosts, asset paths, image, UID/GID, HCA and fabric settings in the copied JSON. Keep the example deployment name, ports and model ID for the commands below. Extract and verify the identical archive at each host's configured bundle path. On the operator machine run:

```sh
python3 portable.py inspect --config ../target-tp2.json --full-cache --output ../inspect-c32.json
python3 portable.py plan --config ../target-tp2.json --inspection ../inspect-c32.json --output-dir ../plan-c32
```

Run the numerical kernel and two-rank RDMA checks in section 2 of the archive README. Copy each generated create script to its corresponding rank and execute it there. Start the resulting containers through the local supervisor, rank 0 followed by rank 1. Wait for model readiness. Keep the candidate endpoint isolated from other traffic.

Sample chat requests for this model are in [examples/c32](examples/c32).

## Run the verified 32 × 32K workload

The default profile has 32 streams, 32,768 context positions per stream and a shared pool of 1,048,576 positions. Open a tunnel in a separate terminal:

```sh
ssh -N -L 18220:127.0.0.1:8220 tf-node0
```

From the extracted bundle on the operator machine, run the following against a fresh candidate. The SSH aliases must allow Docker log and cgroup reads on the respective hosts.

```sh
python3 qualify_c32.py \
  --base-url http://127.0.0.1:18220/v1 \
  --model deepseek-v4.1-flash-1017-directions-c32 \
  --rank0-ssh tf-node0 --rank1-ssh tf-node1 \
  --rank0-container tf-directions-c32-v2-r0 \
  --rank1-container tf-directions-c32-v2-r1 \
  --prompt-tokens 32000 --new-tokens 512 --output-dir ../c32-32k-result
```

Look for exit status 0 and `passed: true` in `measurement.json`. The harness compares four solo modes to 32 simultaneous cold requests, checks packets and output token hashes, and reads rank 0's mixed decode peak from its log. The recorded run reached 32 active decode streams, with at least 31,988 input tokens per request. Both ranks must have monitor coverage across the burst, zero swap usage and zero new OOM events. Use a fresh candidate and output directory for a repeat because the engine logs a mixed peak only when it increases.

## Repeat the larger 32 × 64K run

The recorded 64K test used a separate candidate with context 499,712, pool 15,990,784 and maximum output 32,768. To reproduce its launch settings, start with a separate configuration using context 262,144, pool 8,388,608, maximum output 32,768 and a fresh deployment name. Repeat inspection and plan generation. The archived planner caps context at 262,144. In the generated scripts set `--context 499712` and `TF_DS_WINDOW=15990784` on both ranks, plus `--pool 15990784` on rank 1. Retain the verified binaries and the remaining scheduler settings. Create and qualify the fresh pair as above.

Repeat the same harness command with the new container names, `--prompt-tokens 64000 --new-tokens 2048` and a fresh output directory. At budget 512 an earlier 64K trial reached only 28 mixed decode streams. At budget 2048 the recorded run reached all 32 with a minimum of 63,994 prompt tokens. The budget is an output ceiling. The exact measurements for both passing workloads are in the [reference summary](examples/c32-reference-results.json).
