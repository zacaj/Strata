# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-16 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

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

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 45.8 (42.4-63.3) | 0.75 (0.68-0.97) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1140.8 (1140.2-1144.0) | not measured | 3.56 (3.54-3.58) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1454.6 (1453.3-1462.8) | not measured | 11.11 (11.02-11.12) |
| agent session, follow-up turns | 14540 (14532-14599) at the last turn | see below | up to 64 | 3 | see below | 50.2 (32.7-66.1) | 3.58 (3.20-3.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.7); earlier-turn tokens read again per session: 72 (4-73) (earlier turns that the cache did not hold; the 13730 (13726-13730) new tokens of the files are read once); first turn (nothing cached) first token: 2.36 s (2.23-2.38).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 279.4 | 2.232 | 46.0 |
| 1 | 3754 | 632 | 3122 | 866.6 | 3.665 | 39.6 |
| 2 | 5287 | 3747 | 1540 | 464.2 | 3.357 | 49.0 |
| 3 | 8731 | 5351 | 3380 | 920.1 | 3.73 | 48.6 |
| 4 | 11632 | 8770 | 2862 | 803.6 | 3.608 | 32.7 |
| 5 | 14532 | 11633 | 2899 | 821.1 | 3.59 | 38.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 153 s.
