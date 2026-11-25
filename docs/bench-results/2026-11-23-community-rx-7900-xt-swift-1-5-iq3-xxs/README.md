# Community benchmark on AMD Radeon RX 7900 XT

Measured on 2026-11-23 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT, VRAM not read
- CPU: Intel(R) Core(TM) Ultra 7 265K (20 cores, 20 threads); installed RAM: 125.2 GiB; storage: sda     ST3000DM008-2DM166       sata; sdb     ST3000DM008-2DM166       sata; sdc     Multiple Reader          usb; sdd     WDC WD140EDGZ-11B1PA0    usb; zram0; nvme0n1 Samsung SSD 9100 PRO 2TB nvme; nvme1n1 Samsung SSD 980 PRO 2TB  nvme
- OS: Linux 7.2.11-300.fc45.x86_64; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.66, commit not a git checkout; engine 0.1.44.1
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: swift-1.5-iq3_xxs
- Context: 300072 tokens; parallel slots: 1
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 47.4 (35.5-62.4) | 0.59 (0.52-1.02) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 622.5 (618.8-631.7) | not measured | 6.49 (6.39-6.52) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 673.6 (671.3-680.7) | not measured | 23.88 (23.63-23.96) |
| agent session, follow-up turns | 14836 (14822-14845) at the last turn | see below | up to 64 | 3 | see below | 61.1 (50.1-75.4) | 5.72 (3.35-7.67) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.5 % (70.5-70.7); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 13981 (13980-13982) new tokens of the files are read once); first turn (nothing cached) first token: 2.62 s (2.29-2.65). Corpus: `repo_docs` (7c81d715a005) [README.md, docs/HOW_IT_WORKS.md, docs/DETAILS.md, docs/INSTALL.md, docs/MODELS.md].

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 270.5 | 2.286 | 34.6 |
| 1 | 4910 | 624 | 4286 | 561.0 | 7.67 | 66.0 |
| 2 | 6446 | 4973 | 1473 | 424.9 | 3.506 | 50.9 |
| 3 | 9716 | 6439 | 3277 | 502.6 | 6.557 | 61.6 |
| 4 | 11795 | 9762 | 2033 | 475.2 | 4.311 | 66.0 |
| 5 | 14845 | 11858 | 2987 | 526.3 | 5.725 | 52.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 231 s.
