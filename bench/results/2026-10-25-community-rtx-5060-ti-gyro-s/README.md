# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA: not read
- Strata: project 0.1.44.1, commit b8175ca5; engine 0.1.44.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gyro-s
- Context: 150072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack gyro-s --native Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 150072 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --conversation-cache-mib 5192 --conversation-cache-slots 4
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 49.6 (46.3-51.3) | 0.74 (0.65-0.90) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1179.8 (1174.6-1181.3) | not measured | 3.44 (3.43-3.45) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1563.1 (1551.6-1571.2) | not measured | 10.32 (10.29-10.39) |
| agent session, follow-up turns | 14494 (14493-14515) at the last turn | see below | up to 64 | 3 | see below | 49.6 (24.2-66.1) | 3.46 (3.13-3.62) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.5 % (68.5-68.7); earlier-turn tokens read again per session: 73 (2-73) (earlier turns that the cache did not hold; the 13729 (13728-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.31 s (2.24-2.31).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 277.8 | 2.242 | 38.0 |
| 1 | 3754 | 632 | 3122 | 888.7 | 3.569 | 45.9 |
| 2 | 5287 | 3818 | 1469 | 475.5 | 3.128 | 49.6 |
| 3 | 8731 | 5351 | 3380 | 943.9 | 3.624 | 34.0 |
| 4 | 11594 | 8732 | 2862 | 836.4 | 3.466 | 30.2 |
| 5 | 14494 | 11595 | 2899 | 850.8 | 3.456 | 35.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 143 s.
