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
| short chat | 52 (49-57) | 0 | 100 (22-100) | 9 | not measured | 18.8 (12.4-33.3) | 0.66 (0.61-0.69) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 853.6 (756.4-894.2) | not measured | 4.84 (4.62-5.46) |
| 16K prompt, fresh | 16527 (16527-16527) | 0 (0-0) | 24 cap | 3 | 974.8 (946.4-1078.5) | not measured | 16.97 (15.34-17.48) |
| agent session, follow-up turns | 14265 (14260-14272) at the last turn | see below | up to 64 | 3 | see below | 21.2 (7.9-25.3) | 3.92 (3.05-4.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.6 % (70.5-70.6); earlier-turn tokens read again per session: 1 (1-2) (earlier turns that the cache did not hold; the 13480 (13480-13480) new tokens of the files are read once); first turn (nothing cached) first token: 2.38 s (2.36-2.39). Corpus: `repo_docs` (7c81d715a005) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 523 | 0 | 523 | 220.0 | 2.381 | 22.5 |
| 1 | 4706 | 547 | 4159 | 1031.6 | 4.045 | 23.1 |
| 2 | 6214 | 4763 | 1451 | 477.1 | 3.052 | 23.0 |
| 3 | 9273 | 6267 | 3006 | 750.6 | 4.019 | 17.9 |
| 4 | 11319 | 9336 | 1983 | 573.2 | 3.473 | 14.3 |
| 5 | 14265 | 11383 | 2882 | 744.8 | 3.885 | 23.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 247 s.
