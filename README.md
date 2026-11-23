# Per-request directions for DeepSeek V4.1 Flash on TensorFold TP2

[Download the C32 release assets and reproduce the 32-request benchmark](C32-QUICKSTART.md).

Run one original DeepSeek-V4.1-Flash-1017 EXL3 2.5 bpw checkpoint on two 128 GB GB10s. One chat request can use the baseline, the next can apply a direction at half strength, and the next can return to baseline. The API accepts a strength and an optional layer list. The reference deployment processed one request at a time with eager execution, a 262,144-token context and speculative decoding disabled.

The included patch and launcher reproduce the single-request build (C1). A subsequent C32 build served 32 concurrent requests on another pair. Its peer protocol, strength calibration and measurements are documented in the [C32 field report](C32.md).

## What changes inside the engine

On prefill and decode, each rank has a partial attention output from `wo_b`. Each partial already spans all 5,120 output coordinates. For each token row in a selected layer, the CUDA kernel computes

```text
y ← y − strength × r × dot(r, y)
```

Here `r` is the unit vector for that layer. The transform is linear in `y`, so each rank can apply it before the existing gather using the same strength. The residual stream and hyperconnection branches retain their usual computation. Strength zero skips the kernel. Both ranks receive a 44-byte request descriptor containing the strength, layer mask and direction-bundle hash. TensorFold also includes the descriptor and a weight identifier in its prompt-cache key.

The directions come from the author's published `projected` tensor. Layer selection spans blocks 2 through 39. Strength 1 removes the component along the unit vector up to numerical precision. The relationship between strength and answer behavior is empirical. Reproducing a separately rounded FP8 checkpoint is a different experiment. [Direction source and revision](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/tree/12d58ad0878b5008e80d720251adb969a0069412)

## Starting point

This is a source recipe for an existing working **DeepSeek 1017 EXL3 TP2 deployment**. It uses the GB10 DeepSeek fork at the pinned commit below. Start with its working model view, matching AOT/grammar/vision kit, token map, one prepared rank cache per node, and original native Engram shards 47 and 48. The reference caches are about 93 GB each. Use single-rail RoCE v2 with GID index 3 on both nodes. The reference build used Linux arm64, the fork's Zig 0.17 development toolchain, CUDA 13.0.88 and SM121. [TP2 base recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) · [Pinned original EXL3 model](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw/tree/fdae35e8f1236f0a140e65566894875a6072a73c)

## Build the engine and direction bundle

From this recipe directory, in a Linux build environment with the base recipe's dependencies installed, run:

```bash
recipe=$PWD
git clone https://github.com/bertholomus/TensorFold.git TensorFold
cd TensorFold
git checkout --detach e100ee8b1d806a9f516d7a01cbcbf06ae98d77e1
git apply --check "$recipe/runtime-directions.patch"
git apply "$recipe/runtime-directions.patch"
zig build install native -j2 -Dtarget=aarch64-linux-gnu.2.28 \
  -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121

mkdir -p "$recipe/build/directions"
nvcc -fatbin -O3 --fmad=false --ftz=false -gencode=arch=compute_121,code=sm_121 \
  zig/kernels/cuda/runtime_directions.cu -o "$recipe/build/directions/runtime_directions.fatbin"
nvcc -O3 --fmad=false --ftz=false -arch=sm_121 -Izig/kernels/cuda \
  "$recipe/tests/kernel_check.cu" -o "$recipe/build/kernel_check"
cd "$recipe"
curl -fL -o build/ablation_directions.pt \
  https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/resolve/12d58ad0878b5008e80d720251adb969a0069412/ablation_directions.pt
python3 prepare_directions.py build/ablation_directions.pt build/directions/directions.bin
```

The converter verifies the pinned archive, the selected tensor and the resulting binary. It reads the raw float storage without executing pickle. The expected bundle hash begins with `097fff6bf88d`. The source manifest lists the exact before/after SHA-256 for all 20 patched files, including the fixed-size bundle reader and the reference GID3/cache-builder changes.

## Start the two ranks

On each node, use a separate candidate environment with the same GPU, RDMA and memlock access as the working TP2 baseline. Make the following paths available **inside** it:

| Path | Contents |
| --- | --- |
| `/engine` | Patched `tensorfold-native` and `tf-dsv41-lanes` from `zig-out/bin`, plus the matching baseline `rdma_gather.fatbin` |
| `/directions` | Both files in `build/directions`, identical on both nodes |
| `/model`, `/kit`, `/engram` | Original serving view, baseline kernel kit and native Engram shards, respectively |
| `/cache` | `dsv41_token_map.json` and `rank-cache/rankNof2-*.bin` for this rank |
| `/recipe` | This source recipe, including `reference-weights.json` |

Keep model assets read-only. Set `MASTER` to rank 0's fabric IPv4 on both nodes, and set `HCA` and `IFNAME` to the local RoCE device and network interface on each. The rank checker reads the model view and **the entire local rank cache** and compares them with the reference hashes. Run the numerical gate once on each idle GPU before starting either rank.

```bash
# On each node, with RANK set to 0 or 1
python3 /recipe/verify-rank.py "$RANK"
/recipe/build/kernel_check --execute-on-idle-gpu
```

Start `bash /recipe/launch-rank.sh 0` in the rank 0 environment, then `bash /recipe/launch-rank.sh 1` in the rank 1 environment. Both must use the same `TP_PORT` (default 29535). The API listens on rank 0 loopback port 8220. Use a separate cache namespace if you change the base weights. The included launcher uses a digest of the reference weight manifest and is intended for the verified reference assets.

## Try OFF → ON → OFF

From the recipe directory on rank 0, or through a localhost tunnel forwarded to its port 8220:

```bash
for request in off on-half off; do
  curl -fsS http://127.0.0.1:8220/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary "@examples/$request.json"
  echo
done
python3 qualify.py --base-url http://127.0.0.1:8220/v1 \
  --model deepseek-v4.1-flash-1017-runtime-directions-tp2 \
  --warm-cache --output acceptance.json
```

Omit `ablation` for baseline. The ON example adds the following top-level field to the same request. Omitting `layers` selects all 38 supported layers.

```json
{"ablation":{"enabled":true,"strength":0.5,"direction_set":"pqhaz-1017-projected-experimental","layers":[2,3]}}
```

The qualifier checks the echoed descriptor, deterministic OFF return, streaming, rejection of malformed settings and warm-prefix reuse. It writes full responses to a local file. Identical generated text on a simple prompt can accompany a real logit change. For the reference logit comparison and its limitations, see [the results](RESULTS.md).

## Use the switch in Pi

Install the [Pi extension](https://github.com/anvarazizov/pi-runtime-ablation) once:

```sh
pi install git:github.com/anvarazizov/pi-runtime-ablation@v0.1.0
```

Start Pi with the model ID from your Pi provider entry. For the C32 release:

```sh
ABLATION_MODEL=deepseek-v4.1-flash-1017-directions-c32 pi
```

Select that model through a provider configured for chat completions. For C1 use `deepseek-v4.1-flash-1017-runtime-directions-tp2`, or use the alias your gateway exposes. In the session, use `/ablation half`, `/ablation full` or `/ablation off`. The footer shows the current setting. Leave `ABLATION_SCALE=1` for the kernels in this recipe.

## Credits

TensorFold by Ash Hart and contributors, DeepSeek GB10 port and EXL3 packaging by Bertholomus and contributors, DeepSeek model by DeepSeek-AI, and projected directions by pqhaz. This recipe adds the request controls, TP2 activation hook, cache isolation and verification. TensorFold notices and license texts accompany the patch. Model weights, kernel kits and the author's direction archive are obtained from their respective projects.
