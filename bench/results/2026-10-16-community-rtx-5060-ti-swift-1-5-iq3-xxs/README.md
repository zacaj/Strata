# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA: not read
- Strata: project 0.1.44.1, commit a519aaa4; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 150072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack swift-iq3_xxs --native Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 150072 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --conversation-cache-mib 5192 --conversation-cache-slots 4
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 41.9 (37.4-53.4) | 0.81 (0.71-1.13) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 662.1 (660.4-664.3) | not measured | 6.10 (6.08-6.15) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1047.7 (1047.3-1050.0) | not measured | 15.37 (15.35-15.39) |
| agent session, follow-up turns | 14559 (14548-14581) at the last turn | see below | up to 64 | 3 | see below | 54.6 (28.4-62.1) | 5.68 (5.30-5.98) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13728 (13728-13731) new tokens of the files are read once); first turn (nothing cached) first token: 4.62 s (4.38-4.64).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 139.9 | 4.382 | 35.5 |
| 1 | 3743 | 620 | 3123 | 547.6 | 5.747 | 41.6 |
| 2 | 5276 | 3807 | 1469 | 277.0 | 5.336 | 53.3 |
| 3 | 8720 | 5339 | 3381 | 583.3 | 5.835 | 62.1 |
| 4 | 11646 | 8713 | 2933 | 497.3 | 5.941 | 60.6 |
| 5 | 14581 | 11682 | 2899 | 512.8 | 5.699 | 57.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 219 s.
