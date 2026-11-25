"""Run configs that build on other config files: ``"extends"``.

A run config (``strata-<model>.json``) may name one or more base files::

    {"extends": "configs/swift-iq3_xxs.json", "args": ["--max-context", "65536"], "args_remove": ["--kv-resident"]}

``extends`` is a path relative to the file that names it (or a list of them, later ones winning). Bases may extend
other bases. The resolved config is the bases merged in order, then this file's own keys on top:

* plain keys: this file's value replaces the base's; ``null`` removes the key;
* objects (``env``, ``sampling`` ...): merged key by key, the same rules recursively;
* ``args``: merged by option. A ``--flag`` here replaces every ``--flag`` of the base (with its value) where the base
  had it; options the base does not have are appended. ``args_remove`` lists options of the base to drop.

Writing a resolved config back (setup, calibration, the Settings page) stores only what differs from the bases
(``own_part``), so a tuned value stays in the file that was run and the bases are never rewritten.

A base file must not be named ``strata-*.json`` unless it is a complete run config itself: that pattern is what the
model lists show.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

EXTENDS = "extends"
ARGS_REMOVE = "args_remove"
_MAX_DEPTH = 16


class ConfigLayerError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------------------ args
def _is_flag(tok: Any) -> bool:
    return isinstance(tok, str) and tok.startswith("--")


def _flag_key(tok: str) -> str:
    return tok.partition("=")[0]


def _groups(args: list) -> list[tuple[str, list]]:
    """``args`` as (key, tokens) entries: a ``--flag`` with its value (the next token when it is no ``--`` option),
    ``--flag=value`` alone, and positionals keyed ``#0``, ``#1`` ... in order."""
    out, i, pos = [], 0, 0
    while i < len(args):
        tok = args[i]
        if _is_flag(tok):
            if "=" not in tok and i + 1 < len(args) and not _is_flag(args[i + 1]):
                out.append((_flag_key(tok), [tok, args[i + 1]]))
                i += 2
            else:
                out.append((_flag_key(tok), [tok]))
                i += 1
        else:
            out.append((f"#{pos}", [tok]))
            pos += 1
            i += 1
    return out


def _by_key(groups: list[tuple[str, list]]) -> dict[str, list]:
    """key -> all of its tokens, in order (an option given twice keeps both)."""
    out: dict[str, list] = {}
    for k, toks in groups:
        out.setdefault(k, []).extend(toks)
    return out


def merge_args(base: list, own: list | None, remove: list | None = None) -> list:
    own_groups = _groups(list(own or []))
    own_by = _by_key(own_groups)
    drop = {_flag_key(r) if _is_flag(r) else r for r in (remove or []) if isinstance(r, str)}
    out, done = [], set()
    for k, toks in _groups(list(base or [])):
        if k in own_by:
            if k not in done:
                out.extend(own_by[k])
                done.add(k)
        elif k not in drop:
            out.extend(toks)
    for k, _ in own_groups:
        if k not in done:
            out.extend(own_by[k])
            done.add(k)
    return out


def diff_args(base: list, full: list) -> tuple[list, list]:
    """(the args this file needs, the base options it drops) so that merge_args(base, ...) gives ``full``'s options."""
    b, f = _by_key(_groups(list(base or []))), _groups(list(full or []))
    f_by = _by_key(f)
    own, done = [], set()
    for k, _ in f:
        if k in done:
            continue
        done.add(k)
        if b.get(k) != f_by[k]:
            own.extend(f_by[k])
    return own, [k for k in b if k not in f_by]


# ------------------------------------------------------------------------------------------------------------ dicts
def merge(base: dict, own: dict) -> dict:
    out = json.loads(json.dumps(base))                 # a deep copy: values are JSON
    for k, v in own.items():
        if k in (EXTENDS, ARGS_REMOVE):
            continue
        if v is None:
            out.pop(k, None)
        elif k == "args" and isinstance(v, list):
            out["args"] = merge_args(out.get("args") or [], v, own.get(ARGS_REMOVE))
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = json.loads(json.dumps(v))
    if own.get(ARGS_REMOVE) and "args" not in own:
        out["args"] = merge_args(out.get("args") or [], [], own[ARGS_REMOVE])
    return out


def _diff(base: dict, full: dict, top: bool) -> dict:
    out: dict = {}
    for k, v in full.items():
        if top and k == "args" and isinstance(v, list):
            own, drop = diff_args(base.get("args") or [], v)
            if own:
                out["args"] = own
            if drop:
                out[ARGS_REMOVE] = drop
        elif k not in base:
            out[k] = v
        elif isinstance(v, dict) and isinstance(base[k], dict):
            d = _diff(base[k], v, False)
            if d:
                out[k] = d
        elif v != base[k]:
            out[k] = v
    for k in base:
        if k not in full:
            out[k] = None
    return out


# ------------------------------------------------------------------------------------------------------------ files
def _read_raw(path: Path) -> dict:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as e:
        raise ConfigLayerError(f"cannot read {path}: {e}") from None
    except ValueError as e:
        raise ConfigLayerError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(cfg, dict):
        raise ConfigLayerError(f"{path} is not a JSON object")
    return cfg


def _bases(raw: dict, path: Path) -> list[Path]:
    ext = raw.get(EXTENDS)
    if ext is None:
        return []
    names = [ext] if isinstance(ext, str) else ext
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        raise ConfigLayerError(f'{path.name}: "extends" must be a file name or a list of them')
    return [Path(os.path.expanduser(n)) if os.path.isabs(os.path.expanduser(n)) else path.parent / n for n in names]


def _parents_of(raw: dict, path: Path, chain: tuple) -> dict:
    acc: dict = {}
    for base in _bases(raw, path):
        real = base.resolve()
        if real in chain:
            raise ConfigLayerError(f"{path.name}: \"extends\" loops back to {base}")
        if len(chain) >= _MAX_DEPTH:
            raise ConfigLayerError(f"{path.name}: \"extends\" goes more than {_MAX_DEPTH} files deep")
        acc = merge(acc, _resolve(_read_raw(real), real, chain + (real,)))
    return acc


def _resolve(raw: dict, path: Path, chain: tuple) -> dict:
    return merge(_parents_of(raw, path, chain), raw) if raw.get(EXTENDS) is not None else raw


def has_layers(raw: Any) -> bool:
    return isinstance(raw, dict) and raw.get(EXTENDS) is not None


def resolve(raw: dict, path: str | Path | None) -> dict:
    """``raw`` (a config already parsed from ``path``) with its bases applied; unchanged when it extends nothing."""
    if not has_layers(raw):
        return raw
    if path is None:
        raise ConfigLayerError('a config with "extends" needs its file path')
    p = Path(path).resolve()
    return _resolve(raw, p, (p,))


def load(path: str | Path) -> dict:
    """The run config at ``path``, resolved."""
    return resolve(_read_raw(Path(path)), path)


def own_part(path: str | Path, cfg: dict) -> dict:
    """What to write to ``path`` for the resolved config ``cfg``: ``cfg`` itself when the file on disk extends nothing,
    else ``extends`` and only what differs from the bases."""
    p = Path(path)
    try:
        raw = _read_raw(p) if p.exists() else {}
    except ConfigLayerError:
        return cfg
    if has_layers(cfg) or not has_layers(raw):          # already in file form, or a plain config: written as it is
        return cfg
    ext = raw[EXTENDS]
    parents = _parents_of({EXTENDS: ext}, p.resolve(), (p.resolve(),))
    return {EXTENDS: ext, **_diff(parents, cfg, True)}


def dumps(path: str | Path, cfg: dict, indent: int = 1) -> str:
    """The text to write to ``path`` for the resolved ``cfg`` (see own_part)."""
    return json.dumps(own_part(path, cfg), indent=indent)
