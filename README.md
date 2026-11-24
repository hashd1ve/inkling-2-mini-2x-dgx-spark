# Inkling 2 Mini on 2× DGX Spark

A recipe to serve [**Inkling 2 Mini**](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Thinking Machines Lab,
Apache-2.0) from two NVIDIA DGX Sparks (GB10) linked by their ConnectX ports, with vLLM at TP=2 and three exact patches
on top of the official `vllm/vllm-openai:inkling-2-mini` image.

| Agent steps with tools, 0.2k–142k tokens of context (median tok/s) | Temperature 0 | Temperature 1 |
| --- | --- | --- |
| One Spark, official recipe (DSpark 7) | 54.1 | 44.3 |
| Two Sparks, TP=2, DSpark 7 | 84.7 | 72.9 |
| Two Sparks, this recipe | **101.1** | **79.2** |

Short chat, one user: 66.2 tok/s; 16 users: 332 tok/s in total. Context up to 524,288 tokens per request, with a KV cache
of 2.62M tokens shared by all conversations.

None of the patches changes the weights or the output distribution: perplexity is the same with and without them
(NLL 1.9874 / 1.9087 / 1.0008 on English / Spanish / code).

## What is in it

| | |
| --- | --- |
| [`serve/serve_tp2.sh`](serve/serve_tp2.sh) | Starts vLLM on both Sparks (native multi-node, no Ray) and waits until the API answers |
| [`patch/adaptive/`](patch/adaptive) | `ADAPT=1`. Lets vLLM's adaptive verification run with Inkling's DSpark drafter (the image vetoes it), and adds to its cost model the fixed cost of a step and the cost of attending to the context, which the startup profile does not see. Without them it verifies too few drafts. Rejection sampling stays exact |
| [`patch/rdma/`](patch/rdma) | `RDMA=1`. Replaces NCCL's all-reduce for small bf16 tensors (up to 128 rows) with a direct RDMA exchange over RoCE: 8.7 µs instead of 46 µs for one row, ~96 of them per decoding pass. Same sum as NCCL, bit for bit. Uses one CPU core per Spark while there is traffic |
| [`patch/gemv12/`](patch/gemv12) | `GEMV12=1`. Reads the bf16 weights of attention and the vocabulary head as 12 bits per weight instead of 16 (the exponents fit a window of 15 values; the rest go through escapes). The kernel rebuilds the same bf16 values: output identical bit for bit, 1.23–1.32× faster on those reads. Costs +1.4 GB per Spark |

The files under `patch/` are the image's own files with the changes applied (vLLM, Apache-2.0), mounted over the
originals. They match the `vllm/vllm-openai:inkling-2-mini` image they were taken from; a later image may need them
redone.

## Requirements

- Two DGX Sparks with the ConnectX link up and an IPv4 address on each end (the script assumes interface
  `enp1s0f1np1`, RoCE devices `rocep1s0f1` and `roceP2p1s0f1`, GID index 5 = RoCE v2 with that IPv4).
- The `vllm/vllm-openai:inkling-2-mini` image and the same revision of `thinkingmachines/Inkling-2-Mini` in the
  Hugging Face cache of both (the containers run with `HF_HUB_OFFLINE=1`).
- A machine that reaches both over ssh without a password. The script runs from there.

## Run

Copy the patches to both Sparks:

```bash
for h in <worker> <head>; do ssh $h mkdir -p inkling-tp2; rsync -a patch/ $h:inkling-tp2/; done
```

Start it (about 4 minutes):

```bash
export HEAD=<head> HEAD_IP=<head-link-ip> WORKER=<worker> WORKER_IP=<worker-link-ip>
SPEC=7 ADAPT=1 RDMA=1 GEMV12=1 bash serve/serve_tp2.sh
```

The OpenAI-compatible API is on the head node, port 8010. `STOP=1 bash serve/serve_tp2.sh` stops it
(with `HEAD` and `WORKER` set).

| Variable | Default | |
| --- | --- | --- |
| `HEAD`, `HEAD_IP` | required | ssh name and ConnectX link IPv4 of the node with the API |
| `WORKER`, `WORKER_IP` | required | The same for the other node |
| `PORT` | `8010` | API port on the head node |
| `SPEC` | `7` | DSpark draft tokens. Only 7 works (the block size it was trained with); `0` turns it off |
| `MAXLEN` | `524288` | Longest context of a request (the model takes up to 1,048,576) |
| `MEM` | `0.75` | Share of memory for vLLM. Leaves 8–10 GB free per Spark under load; 0.8 is too tight |
| `ADAPT_FIXED_MS`, `ADAPT_CTX_MS_PER_1K` | `15`, `0.09` | The two costs added to the adaptive verification's model |
| `RDMA_AR_DEV`, `RDMA_AR_GID` | `rocep1s0f1`, `5` | RoCE device and GID index for the RDMA exchange (pass them with `EXTRA_ENV="-e ..."`) |

With other interface or device names, change the `NCCL=` line of the script.

## Notes

- If one node goes down, the other waits for it forever (as with NCCL): restart both containers.
- The RDMA library is built with the image's `gcc` when the server starts.
- The server is not deterministic even at temperature 0 (bf16 rounding changes with the batch shape), so outputs
  cannot be compared token by token between runs.
- Prefill runs at ~3,600 tok/s on average (4,650 with a short context, 2,550 at the end of 142k tokens) and is a single
  queue for all requests.

## License

Apache-2.0, as vLLM and the model.
