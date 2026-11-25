# Community benchmark on AMD Radeon RX 7900 XT

Measured on 2026-11-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT, 20 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 265K (20 cores, 20 threads); installed RAM: 125.2 GiB; storage: sda     ST3000DM008-2DM166       sata; sdb     ST3000DM008-2DM166       sata; sdc     Multiple Reader          usb; sdd     WDC WD140EDGZ-11B1PA0    usb; zram0; nvme0n1 Samsung SSD 9100 PRO 2TB nvme; nvme1n1 Samsung SSD 980 PRO 2TB  nvme
- OS: Linux 7.2.11-300.fc45.x86_64; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.66, commit not a git checkout; engine 0.1.66
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-gyro-l
- Context: 150072 tokens; parallel slots: 2
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 32.3 (28.2-42.7) | 0.98 (0.84-1.15) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1052.5 (1051.8-1085.0) | not measured | 3.85 (3.75-3.85) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1357.6 (1245.9-1462.8) | not measured | 11.86 (11.01-12.92) |
| agent session, follow-up turns | 14854 (14830-14859) at the last turn | see below | up to 64 | 3 | see below | 46.5 (18.5-57.3) | 4.46 (3.42-15.80) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.7 % (70.7-70.7); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13983 (13983-13983) new tokens of the files are read once); first turn (nothing cached) first token: 2.98 s (2.96-11.93). Corpus: `repo_docs` (7c81d715a005) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 51.2 | 11.933 | 33.1 |
| 1 | 4926 | 638 | 4288 | 363.6 | 11.826 | 18.5 |
| 2 | 6462 | 4989 | 1473 | 414.3 | 3.59 | 40.1 |
| 3 | 9732 | 6525 | 3207 | 492.8 | 6.542 | 48.1 |
| 4 | 11829 | 9795 | 2034 | 420.0 | 4.877 | 46.5 |
| 5 | 14854 | 11868 | 2986 | 674.4 | 4.458 | 28.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 223 s.
