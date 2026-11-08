# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA: not read
- Strata: project 0.1.51, commit bdc2d88e; engine 0.1.51
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gyro-coder-l
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack gyro-coder-l --native Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf --expert-profile expert-profile-gyro-coder-l.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97 --conversation-cache-mib 5192 --conversation-cache-slots 4
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 25.4 (20.1-27.5) | 1.48 (1.28-1.65) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 982.3 (963.8-984.3) | not measured | 4.14 (4.14-4.23) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1404.7 (1402.8-1411.5) | not measured | 11.51 (11.44-11.51) |
| agent session, follow-up turns | 12724 (7941-12746) at the last turn | see below | up to 64 | 3 | see below | 32.0 (18.3-40.6) | 4.48 (3.90-4.86) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.0 % (54.5-69.2); earlier-turn tokens read again per session: 3 (2-66) (earlier turns that the cache did not hold; the 11894 (7227-11895) new tokens of the files are read once); first turn (nothing cached) first token: 3.11 s (2.92-3.18).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 211.5 | 2.924 | 24.0 |
| 1 | 3212 | 634 | 2578 | 583.0 | 4.476 | 25.5 |
| 2 | 4723 | 3254 | 1469 | 375.9 | 3.97 | 32.0 |
| 3 | 7962 | 4783 | 3179 | 665.1 | 4.855 | 32.8 |
| 4 | 9985 | 7955 | 2030 | 457.4 | 4.486 | 32.0 |
| 5 | 12746 | 10042 | 2704 | 601.6 | 4.568 | 27.6 |

### Skipped and failed

- failed agent: no token came back

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 205 s.
