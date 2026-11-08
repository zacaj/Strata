# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-10-25 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA: not read
- Strata: project 0.1.51, commit bdc2d88e; engine 0.1.51
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 52.2 (47.3-54.9) | 0.69 (0.58-0.83) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1202.4 (1193.8-1204.9) | not measured | 3.37 (3.37-3.40) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1572.7 (1559.9-1585.8) | not measured | 10.26 (10.18-10.34) |
| agent session, follow-up turns | 12722 (12699-12747) at the last turn | see below | up to 64 | 3 | see below | 74.5 (54.7-82.4) | 3.31 (3.05-3.79) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.0-69.2); earlier-turn tokens read again per session: 74 (3-74) (earlier turns that the cache did not hold; the 11896 (11895-11897) new tokens of the files are read once); first turn (nothing cached) first token: 2.25 s (2.22-2.27).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 278.9 | 2.219 | 34.9 |
| 1 | 3205 | 627 | 2578 | 779.2 | 3.364 | 54.7 |
| 2 | 4738 | 3268 | 1470 | 478.3 | 3.113 | 55.4 |
| 3 | 7982 | 4731 | 3251 | 867.3 | 3.79 | 74.9 |
| 4 | 9973 | 8009 | 1964 | 614.2 | 3.233 | 74.5 |
| 5 | 12699 | 9993 | 2706 | 803.6 | 3.407 | 62.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 142 s.
