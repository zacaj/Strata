# Community benchmark on NVIDIA GeForce RTX 5060 Ti

Measured on 2026-11-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti, 16 GiB VRAM
- CPU: AMD Ryzen 7 2700X Eight-Core Processor (8 cores, 16 threads); installed RAM: 64.0 GiB; storage: PC801 NVMe SK hynix 2TB SSD/NVMe; Seagate Portable Unspecified/USB; WD Elements 25A3 HDD/USB; WD Elements 25A3 HDD/USB
- OS: Windows 10 (10.0.19045); GPU driver: 610.62; CUDA (driver supports up to): 13.3
- Strata: project 0.1.62, commit not a git checkout; engine 0.1.62
- GPU detail (after the run, near idle): power_limit_w=180.00, power_draw_w=83.66, pcie=gen3 x8, pstate=P1, temp_c=60
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: gyro-m
- Context: 300000 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata.exe --pack gyro-m --native Qwen3.8-Flash-Next-Gyro-M-TQ2_0-00001-of-00003.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp rt --max-context 300000 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --vision --conversation-cache-mib 10000 --conversation-cache-slots 4
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 40.4 (31.4-43.0) | 0.82 (0.71-1.05) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 831.0 (830.7-832.4) | not measured | 4.87 (4.87-4.88) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1588.5 (1583.4-1613.0) | not measured | 10.17 (10.03-10.21) |
| agent session, follow-up turns | 9091 (9091-9092) at the last turn | see below | up to 64 | 3 | see below | 52.0 (26.9-75.1) | 3.78 (2.90-4.60) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 71.1 % (71.1-71.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8208 (8207-8209) new tokens of the files are read once); first turn (nothing cached) first token: 4.31 s (4.31-4.73).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 142.7 | 4.312 | 23.2 |
| 1 | 2268 | 626 | 1642 | 566.5 | 2.927 | 37.2 |
| 2 | 3974 | 2331 | 1643 | 524.4 | 3.19 | 57.3 |
| 3 | 5680 | 4037 | 1643 | 438.0 | 3.804 | 75.1 |
| 4 | 7386 | 5743 | 1643 | 417.2 | 3.989 | 65.7 |
| 5 | 9092 | 7449 | 1643 | 415.4 | 3.987 | 63.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 185 s.
