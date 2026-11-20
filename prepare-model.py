"""Packs a GGUF, writes strata-<name>.json (and trace-<name>.json), then offers to calibrate.

Usage: prepare-model.py <shard-1.gguf> [extra iq_pack.py options, e.g. --compat-bf16]
It asks for a short name, suggesting one from the file name. If the pack step says to use --compat-bf16, it offers to retry with it.
"""
import json, os, re, shutil, struct, subprocess, sys
from pathlib import Path

ROOT = Path(r"C:\programs\Strata")
PACKS = Path(r"C:\programs\Strata-data\packs")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
TEMPLATE = ROOT / "model-template.json"  # settings every generated config starts from; not strata-*.json, which START-HERE lists as models
SPEC_FLAGS = ("--spec", "--mtp", "--spec-min-p")   # speculative decoding: a native pack needs it


def run(*cmd, capture=False):
    print(">", " ".join(cmd))
    if not capture:
        if subprocess.call(cmd, cwd=ROOT) != 0:
            sys.exit("failed")
        return 0, ""
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        print(line, end="")
        out.append(line)
    return p.wait(), "".join(out)


def profile_shape(path):
    """(layers, experts) from a profile's STRP header."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:4] != b"STRP":
        return None
    _, layers, experts = struct.unpack("<3I", head[4:16])
    return layers, experts


def ask(q):
    return input(q + " [y/N] ").strip().lower().startswith("y")


def write_run_bat(name, serve_json):
    # A run-<name>.bat that serves it, like the other run-*.bat files. CRLF: cmd misreads a batch file without it.
    run_bat = ROOT / f"run-{name}.bat"
    if not run_bat.exists():
        cfg = json.loads(serve_json.read_text())
        run_bat.write_text(
            "@echo off\n"
            f"title Strata {name}\n"
            f'cd /d "{ROOT}"\n'
            f'"{PY}" "{ROOT / "serve" / "server.py"}" "--engine" "strata" "--config" "{serve_json}" '
            f'"--port" "{cfg.get("port", 8080)}" "--host" "{cfg.get("host", "127.0.0.1")}"\n'
            "if errorlevel 1 pause\n", newline="\r\n")
        print(f"Wrote {run_bat.name}")


def offer_calibration(serve_json):
    # Calibration: measures the hardware-dependent settings on this PC and writes the fastest into the config.
    if ask("Calibrate the settings for this PC now (~15-30 min, server must be stopped)?"):
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg = json.loads(serve_json.read_text())
        res = CAL.run(cfg)
        cfg["args"] = CAL.apply(cfg["args"], res["settings"])
        if "env" in res:
            cfg["env"] = CAL.apply_env(cfg.get("env"), res["env"])
        serve_json.write_text(json.dumps(cfg, indent=2) + "\n")
        print("Calibrated:", res["settings"] or "the template's settings are already the fastest",
              f"({res['report'].get('tok_s')} tok/s)" if res["report"].get("tok_s") else "")


def prepare_a3b(gguf, name):
    """Qwen3.6-35B-A3B (GGUF architecture qwen35moe): the engine reads the GGUF as it is and places and learns its
    experts itself, so there is no pack, no expert profile and no trace config; only the config, the run script and
    (optionally) the calibration."""
    # The tokenizer is the Qwen3 one every qwen36 install shares: borrow the folder of one that is already set up.
    tokenizers = sorted(PACKS.glob("qwen36*/tokenizer")) or sorted(PACKS.glob("*/tokenizer"))
    if not tokenizers:
        sys.exit(f"no tokenizer folder under {PACKS} to borrow: install any Qwen3.6 model once (strata launch --family qwen36)")
    tokenizer = tokenizers[0]
    serve_json = ROOT / f"strata-{name}.json"
    if serve_json.exists() and not ask(f"{serve_json.name} exists. Rewrite it from the template "
                                       "(loses its calibrated and hand-edited settings)?"):
        print(f"Kept {serve_json.name}")
    else:
        c = json.loads(TEMPLATE.read_text())
        c["args"] = ["--native", str(gguf), "--max-context", "32768", "--prefill", "auto", "--mtp", "2"]
        c["tokenizer"] = str(tokenizer)
        c["model_name"] = name
        c["arch"] = "qwen35moe"
        c["log"] = str(ROOT / f"strata-{name}.log")
        c["exe"] = str(ROOT / "build" / "strata.exe")
        c.pop("sampling", None)          # the template's are Flash-Next's; the server has this family's own
        serve_json.write_text(json.dumps(c, indent=2) + "\n")
        print(f"Wrote {serve_json.name} (tokenizer: {tokenizer})")
    write_run_bat(name, serve_json)
    print("A3B model: no pack and no expert profile are needed; the engine places experts and adapts as it runs.")
    offer_calibration(serve_json)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    gguf = Path(args.pop(0)).resolve()
    if not gguf.is_file():
        sys.exit(f"not found: {gguf}")
    pack_opts = args
    sys.path.insert(0, str(ROOT / "tools"))
    import gguf_reader
    arch = gguf_reader.GGUFFile(gguf).metadata["general.architecture"]
    a3b = arch == "qwen35moe"
    suggest = re.sub(r"-0000\d-of-\d+$", "", gguf.stem)
    suggest = re.sub(r"^Qwen[\d.]+-Flash-Next-", "", suggest).lower()
    if a3b:
        suggest = re.sub(r"^qwen3\.6-35b-a3b-", "qwen36-", suggest.lower())
    name = input(f"Short name for the {'config' if a3b else 'pack'} and config files [{suggest}]: ").strip() or suggest
    if not re.fullmatch(r"[\w.][\w.-]*", name):
        sys.exit(f"bad name: {name!r}")
    if a3b:
        return prepare_a3b(gguf, name)
    pack = PACKS / name
    profile = ROOT / "data" / f"expert-profile-{name}.bin"
    traces = ROOT / f"traces-{name}"

    # The shard that carries the PLE table: shard 2 in ISTA's files, shard 1 in Groot's.
    ple = None
    m = re.match(r"(.*-)00001-of-(\d+)\.gguf$", gguf.name)
    if m:
        sys.path.insert(0, str(ROOT / "tools"))
        import gguf_reader
        for i in range(1, int(m.group(2)) + 1):
            cand = gguf.with_name(f"{m.group(1)}{i:05d}-of-{m.group(2)}.gguf")
            if cand.exists() and gguf_reader.GGUFFile(cand).find("per_layer_token_embd"):
                ple = cand
                break


    if (pack / "tokenizer").exists() and not ask(f"{pack} exists. Repack?"):
        pass
    else:
        os.environ["STRATA_GGUF_PY"] = r"C:\programs\agention-llama.cpp\gguf-py"
        cmd = [PY, r"tools\iq_pack.py", "--gguf", str(gguf), "--out", str(pack), *pack_opts]
        rc, out = run(*cmd, capture=True)
        if rc and "--compat-bf16" in out and "--compat-bf16" not in pack_opts and ask("The pack step needs --compat-bf16 (dequantizes those small projections to BF16). Retry with it?"):
            rc, out = run(*cmd, "--compat-bf16", capture=True)
        if rc:
            sys.exit("pack failed")
        if traces.exists():                            # recorded on the old pack
            shutil.rmtree(traces)

    def set_profile(a, prof):
        # A native pack needs --spec, and --spec needs a profile whose layers x experts match the model.
        tmpl = json.loads(TEMPLATE.read_text())["args"]
        if "--expert-profile" in a:
            i = a.index("--expert-profile")
            del a[i:i + 2]
        i = a.index("--native") + 2
        a[i:i] = ["--expert-profile", str(prof)]
        for flag in SPEC_FLAGS:                        # only the missing ones: keep a calibrated --spec-min-p
            if flag in tmpl and flag not in a:
                a += [flag, tmpl[tmpl.index(flag) + 1]]

    def config(prof, log, serve):
        c = json.loads(TEMPLATE.read_text())
        a = c["args"]
        def put(flag, val):
            if flag in a:
                a[a.index(flag) + 1] = val
            else:
                i = a.index("--native") + 2
                a[i:i] = [flag, val]
        put("--pack", str(pack))
        put("--native", str(gguf))
        if ple:
            put("--ple-gguf", str(ple))
        set_profile(a, prof)
        c["exe"] = str(ROOT / "build" / "strata.exe")   # the one build: build-strata.bat (and setup) build here
        c["tokenizer"] = str(pack / "tokenizer")
        c["model_name"] = name
        c["log"] = str(ROOT / log)
        if not serve:
            c.pop("host", None)
        return c

    # Until this model has its own profile, borrow one with the same layers x experts.
    sys.path.insert(0, str(ROOT / "tools"))
    import gguf_reader
    meta = gguf_reader.GGUFFile(gguf).metadata
    arch = meta["general.architecture"]
    shape = (meta[f"{arch}.block_count"], meta[f"{arch}.expert_count"])
    have_profile = profile.exists()
    stand_in = None
    for cand in sorted((ROOT / "data").glob("expert-profile*.bin")):
        if cand != profile and profile_shape(cand) == shape:
            stand_in = cand
            break
    if not have_profile and not stand_in:
        sys.exit(f"no profile in the data folder fits {shape[0]} layers x {shape[1]} experts to start from")
    current = profile if have_profile else stand_in

    serve_json = ROOT / f"strata-{name}.json"
    trace_json = ROOT / f"trace-{name}.json"            # also kept out of START-HERE's list
    # An existing config may hold calibrated or hand-tuned settings: keep it unless asked.
    if not serve_json.exists() or ask(f"{serve_json.name} exists. Rewrite it from the template "
                                      "(loses its calibrated and hand-edited settings)?"):
        serve_json.write_text(json.dumps(config(current, f"strata-{name}.log", True), indent=2) + "\n")
        print(f"Wrote {serve_json.name}")
    else:
        cfg = json.loads(serve_json.read_text())
        set_profile(cfg["args"], current)
        serve_json.write_text(json.dumps(cfg, indent=2) + "\n")
    trace_json.write_text(json.dumps(config(stand_in or profile, f"strata-{name}-trace.log", False),
                                     indent=2) + "\n")

    write_run_bat(name, serve_json)

    # 1. The expert profile: which experts this model uses most, so the hot ones start in VRAM.
    if ask("Build the expert profile now (~30 min, server must be stopped)?"):
        run(PY, r"tools\collect_profile_traces.py", str(trace_json), str(traces), "--mode", "routing")
        parts = [f"{t}:1" for t in sorted(traces.glob("c*.trace"))]
        run(PY, r"tools\make_profile_mix.py", *parts, "--n-expert", str(shape[1]), "--out", str(profile))
        cfg = json.loads(serve_json.read_text())
        set_profile(cfg["args"], profile)
        serve_json.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"Done: {profile} (now used by {serve_json.name})")
    elif not have_profile:
        print(f"{serve_json.name} uses {stand_in.name} (another model's) until you build its own.")

    # 2. Calibration
    offer_calibration(serve_json)


if __name__ == "__main__":
    main()
