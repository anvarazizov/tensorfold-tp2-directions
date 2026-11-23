#!/usr/bin/env bash
# Run inside the prepared CUDA environment on each host.
set -euo pipefail
: "${MASTER:?set MASTER to rank 0 fabric IPv4}" "${HCA:?set HCA to this rank RDMA device}"
: "${IFNAME:?set IFNAME to this rank fabric interface}"
rank=${1:?usage: launch-rank.sh 0 or 1}
case "$rank" in 0|1) ;; *) exit 2 ;; esac
here=$(cd "$(dirname "$0")" && pwd)
export TF_RUNTIME_DIRECTIONS_EXPERIMENTAL=1
export TF_RUNTIME_DIRECTIONS_BUNDLE=/directions/directions.bin
export TF_RUNTIME_DIRECTIONS_KERNEL=/directions/runtime_directions.fatbin
export TF_RUNTIME_DIRECTIONS_KERNEL_SHA256=$(sha256sum "$TF_RUNTIME_DIRECTIONS_KERNEL" | cut -d' ' -f1)
export TF_RUNTIME_DIRECTIONS_SHA256=097fff6bf88de26730240d1bc20747fb63cc488db050807a6e8efce7452d763c
export TF_RUNTIME_DIRECTIONS_WEIGHTS_SHA256=$(sha256sum "$here/reference-weights.json" | cut -d' ' -f1)
export TF_RUNTIME_DIRECTIONS_MASK=1099511627772 TF_RUNTIME_DIRECTIONS_SET=pqhaz-1017-projected-experimental
export TF_DS_GRAPHS=0 TF_DS_HC_SIDE=0 TF_DS_STREAMS=1 TF_DS_WINDOW=1048576 TF_DS_ROUND_ROWS=16
export TF_DS_L2_PREFETCH=1 TF_DS_ENGRAM_AIO=1 TF_DS_IDX_SPLIT=8192 TF_DS_REPLAY_FLOOR=1 TF_DS_PROMPT_ALIAS=1
export TF_RDMA_DEVICES="$HCA" TF_DS_RDMA_KERNELS=/engine/rdma_gather.fatbin
export NCCL_IB_HCA="$HCA" NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME="$IFNAME" NCCL_IB_DISABLE=0 NCCL_CROSS_NIC=0 NCCL_IB_MERGE_NICS=0
export TENSORFOLD_PLAIN_SPECIALS=1 TENSORFOLD_GRAMMAR_LIB=/kit/grammar/libtfgrammar.so
port=${TP_PORT:-29535}
if [[ $rank == 0 ]]; then
  export TF_TP_WORLD=2 TF_TP_MASTER="$MASTER" TF_TP_PORT="$port"
  export TF_DS_RANK_CACHE=/cache/rank-cache TF_DS_KIT=/kit TENSORFOLD_CUDA_KERNELS=/kit/aot
  export TF_DS_ENGRAM=/engram TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json
  exec /engine/tensorfold-native serve /model --host 127.0.0.1 --port 8220 \
    --name deepseek-v4.1-flash-1017-runtime-directions-tp2 --context 262144 --parallel 1 \
    --max-tokens 16384 --prompt-cache-gib 1 --no-drafts --backend cuda
fi
exec /engine/tf-dsv41-lanes /model /cache/rank-cache 1 2 "$MASTER" "$port" /kit \
  --pool 1048576 --context 262144 --streams 1 --round-rows 16 --engram /engram \
  --token-map /cache/dsv41_token_map.json --graphs 0 --side 0 --prefetch 1 --aio 1 \
  --rdma "$HCA" --rdma-kernels "$TF_DS_RDMA_KERNELS" --drafts 0
