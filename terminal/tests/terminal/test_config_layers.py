"""Run configs with "extends": merging, args by option, writing back only what differs, and the readers that resolve them."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from strata_control import config_layers as L

BASE = {"exe": "component:strata-engine", "model_name": "swift", "env": {"A": "1", "B": "2"},
        "args": ["--pack", "P", "--native", "N", "--kv-resident", "32768", "--spec", "4", "--flagonly", "--max-context", "300072"]}


class LayerCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="strata-layers-")
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "configs").mkdir()
        self.write("configs/swift.json", BASE)

    def write(self, name, cfg):
        p = self.dir / name
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return p


class MergeTests(LayerCase):
    def test_plain_config_is_unchanged(self):
        p = self.write("strata-a.json", BASE)
        self.assertEqual(L.load(p), BASE)
        self.assertIs(L.own_part(p, BASE), BASE)

    def test_args_replace_by_option_in_place_and_append_new(self):
        p = self.write("strata-a.json", {"extends": "configs/swift.json", "args": ["--max-context", "65536", "--spec", "6", "--new", "v"]})
        self.assertEqual(L.load(p)["args"], ["--pack", "P", "--native", "N", "--kv-resident", "32768", "--spec", "6",
                                             "--flagonly", "--max-context", "65536", "--new", "v"])

    def test_args_remove_and_null_keys(self):
        p = self.write("strata-a.json", {"extends": "configs/swift.json", "args_remove": ["--kv-resident", "--flagonly"],
                                         "env": {"B": None, "C": "3"}, "model_name": None})
        cfg = L.load(p)
        self.assertEqual(cfg["args"], ["--pack", "P", "--native", "N", "--spec", "4", "--max-context", "300072"])
        self.assertEqual(cfg["env"], {"A": "1", "C": "3"})
        self.assertNotIn("model_name", cfg)
        self.assertNotIn("extends", cfg)

    def test_chain_and_list_later_wins(self):
        self.write("configs/hw.json", {"extends": "swift.json", "args": ["--spec", "8"], "port": 8081})
        self.write("configs/other.json", {"port": 9000})
        p = self.write("strata-a.json", {"extends": ["configs/hw.json", "configs/other.json"]})
        cfg = L.load(p)
        self.assertEqual(cfg["port"], 9000)
        self.assertIn("8", cfg["args"])

    def test_loop_is_refused(self):
        self.write("configs/x.json", {"extends": "y.json"})
        self.write("configs/y.json", {"extends": "x.json"})
        p = self.write("strata-a.json", {"extends": "configs/x.json"})
        with self.assertRaises(L.ConfigLayerError):
            L.load(p)


class WriteBackTests(LayerCase):
    def test_round_trip_keeps_only_the_difference(self):
        p = self.write("strata-a.json", {"extends": "configs/swift.json", "args": ["--max-context", "65536"]})
        cfg = L.load(p)
        cfg["args"][cfg["args"].index("--spec") + 1] = "6"
        cfg["args"].remove("--flagonly")
        cfg["port"] = 9000
        cfg["env"]["A"] = "9"
        own = L.own_part(p, cfg)
        self.assertEqual(own["extends"], "configs/swift.json")
        self.assertEqual(own["args"], ["--spec", "6", "--max-context", "65536"])
        self.assertEqual(own["args_remove"], ["--flagonly"])
        self.assertEqual(own["env"], {"A": "9"})
        self.assertNotIn("exe", own)
        p.write_text(L.dumps(p, cfg), encoding="utf-8")
        self.assertEqual(L.load(p), cfg)
        self.assertEqual(json.loads((self.dir / "configs/swift.json").read_text()), BASE)   # the base is never written

    def test_file_form_is_written_as_given(self):
        p = self.write("strata-a.json", {"extends": "configs/swift.json"})
        raw = {"extends": "configs/swift.json", "args_remove": ["--spec"]}
        self.assertEqual(L.own_part(p, raw), raw)


class ReaderTests(LayerCase):
    def test_server_and_runconfig_resolve(self):
        from serve import runconfig
        from serve.server_parts.http_api import read_config
        p = self.write("strata-a.json", {"extends": "configs/swift.json", "port": 9000})
        self.assertEqual(read_config(p)["exe"], "component:strata-engine")
        cfg = runconfig.load(p)
        cfg["port"] = 9001
        runconfig.save(p, cfg)
        self.assertEqual(json.loads(p.read_text()), {"extends": "configs/swift.json", "port": 9001})

    def test_profiles_reader_resolves(self):
        from strata_control import profiles
        p = self.write("strata-a.json", {"extends": "configs/swift.json", "tokenizer": "tok"})
        self.assertEqual(profiles.read_run_config(p)["model_name"], "swift")


if __name__ == "__main__":
    unittest.main()
