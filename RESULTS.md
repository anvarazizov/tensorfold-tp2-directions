# Reference results

Measured on 16 November 2026 with the original 2.5 bpw checkpoint on two GB10s. Both modes used the same compiled eager engine. Each logit probe used a distinct cache salt to force a cold prefill. For the prompt “Return only the integer: 31 * 17.” the hashes of the full FP32 prefill logits were:

| Request | SHA-256 |
| --- | --- |
| Field omitted | `8ce7aa22491c7aeac39d3d411b3dab6557132780422922cb623dd6c80aad5e0a` |
| Explicit zero | `8ce7aa22491c7aeac39d3d411b3dab6557132780422922cb623dd6c80aad5e0a` |
| Strength 0.5 | `9c1ca3d2e9900d2bff445f58a215faac0d8fe72cdf614eb583529794dda2d53b` |
| Field omitted again | `8ce7aa22491c7aeac39d3d411b3dab6557132780422922cb623dd6c80aad5e0a` |

The baseline returned exactly after the half-strength request. The changed logits establish that the intervention reached the model. These measurements cover the pinned TP2 layout with one active request at a time.

| Check | Observed result |
| --- | --- |
| Independent CUDA numerical checks | 16 cases, maximum recorded error 1.35 × 10⁻⁷ or less |
| Bounded functional suite | 18/18 at baseline and 18/18 at strength 0.5, including 250K retrieval, 150K coding, tools, vision and queued requests |
| Strict API suite | 40/43. All 27 engineering checks passed. Three Ukrainian arithmetic answers included extra words or a calculation alongside the correct integer |
| Client integration | 6/6 Pi SDK checks and 4/4 OpenWebUI provider checks |
| Median short decode rate, three probes per setting | 49.38 tokens/s at baseline and 48.33 tokens/s at 0.5 |

The short throughput sample does not resolve a general performance cost. The evaluation measured functional behavior and cache isolation. Refusal rates, mixed-mode concurrent batches, CUDA graphs, speculative decoding, TP4 and TP8 fall outside these results. The GPU results come from the original deployment. The source-only distribution was subsequently checked offline against that build.

## Repeat the logit check

Start the patched rank 0 with `TF_RUNTIME_LOGITS_SHA=1` in its environment and capture stderr. On an otherwise idle candidate, issue these four requests in order:

```bash
python3 tests/probe_logits.py http://127.0.0.1:8220/v1
```

Read the four new JSON records with the `prompt_logits` field from rank 0 stderr. Digests one, two and four should match, and digest three should differ. Compare within the same engine build first. Compiler or baseline differences can change the absolute digests. Each call uses a fresh cache salt and requests one output token.

The subsequent 32-stream build has a separate [C32 field report and paired behavioral results](C32.md).
