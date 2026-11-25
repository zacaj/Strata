# Community benchmark on AMD Radeon RX 7900 XT

Measured on 2026-11-24 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT, 20 GiB VRAM
- CPU: Intel(R) Core(TM) Ultra 7 265K (20 cores, 20 threads); installed RAM: 125.2 GiB; storage: sda     ST3000DM008-2DM166       sata; sdb     ST3000DM008-2DM166       sata; sdc     Multiple Reader          usb; sdd     WDC WD140EDGZ-11B1PA0    usb; zram0; nvme0n1 Samsung SSD 9100 PRO 2TB nvme; nvme1n1 Samsung SSD 980 PRO 2TB  nvme
- OS: Linux 7.2.11-300.fc45.x86_64; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.66, commit not a git checkout; engine hy4-serve-0
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: hy4-flash-sq-64
- Context: 65536 tokens; parallel slots: 1
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
| short chat | 52 (49-57) | 0 | 100 (22-100) | 9 | not measured | 22.5 (16.5-32.1) | 0.62 (0.56-0.75) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 471.4 (471.0-478.8) | not measured | 8.76 (8.62-8.76) |
| 16K prompt, fresh | 16527 (16527-16527) | 0 (0-0) | 24 cap | 3 | 435.8 (430.4-442.1) | not measured | 37.94 (37.40-38.41) |
| agent session, follow-up turns | 14266 (14253-14267) at the last turn | see below | up to 64 | 3 | see below | 21.6 (15.7-26.7) | 7.91 (5.06-9.75) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.6 % (70.5-70.6); earlier-turn tokens read again per session: 1 (1-1) (earlier turns that the cache did not hold; the 13480 (13480-13480) new tokens of the files are read once); first turn (nothing cached) first token: 2.80 s (2.77-2.80). Corpus: `repo_docs` (7c81d715a005) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 523 | 0 | 523 | 188.9 | 2.773 | 14.9 |
| 1 | 4706 | 547 | 4159 | 430.1 | 9.684 | 18.2 |
| 2 | 6216 | 4765 | 1451 | 280.6 | 5.181 | 17.9 |
| 3 | 9275 | 6269 | 3006 | 363.6 | 8.28 | 18.6 |
| 4 | 11321 | 9338 | 1983 | 325.1 | 6.113 | 15.7 |
| 5 | 14267 | 11385 | 2882 | 365.2 | 7.907 | 19.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 359 s.
