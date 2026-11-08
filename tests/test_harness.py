"""Harness tests. Run: venv/bin/python -m unittest tests.test_harness -v   (or pytest, if installed).

The live API smoke test is skipped without ANTHROPIC_API_KEY (or with SKIP_LIVE=1).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import logging_utils as lu  # noqa: E402
from harness.claude_client import ask, has_api_key  # noqa: E402
from harness.sandbox import Sandbox  # noqa: E402
from harness.weights_text import (diff_params, params_from_state_dict, params_to_state_dict,  # noqa: E402
                                  parse_weights, weights_to_text)


def mlp(*arch: int, seed: int = 0) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = []
    for i, (a, b) in enumerate(zip(arch[:-1], arch[1:])):
        layers += [torch.nn.Linear(a, b)] + ([torch.nn.Tanh()] if i < len(arch) - 2 else [])
    return torch.nn.Sequential(*layers)


class TestWeightsText(unittest.TestCase):
    def setUp(self) -> None:
        self.net = mlp(8, 6, 4)
        self.sd = self.net.state_dict()
        self.p = params_from_state_dict(self.sd)

    def test_format(self) -> None:
        t = weights_to_text(self.sd, header=False)
        lines = t.splitlines()
        self.assertEqual(len(lines), 8 + 1 + 6 + 1)
        self.assertTrue(lines[0].startswith("W1 row 0  "))
        self.assertEqual(len(lines[0].split()) - 3, 6)          # W printed as [in, out]
        self.assertTrue(lines[8].startswith("b1  "))
        self.assertNotIn("-0.000", t)
        w0 = self.sd["0.weight"][:, 0].numpy()                   # input unit 0 -> all hidden units
        self.assertAlmostEqual(float(lines[0].split()[3]), float(w0[0]), places=3)

    def test_roundtrip(self) -> None:
        back = parse_weights(weights_to_text(self.sd))
        for k in self.p:
            np.testing.assert_allclose(back[k], self.p[k], atol=5e-4 + 1e-7)
        sd2 = params_to_state_dict(back, self.sd)
        self.net.load_state_dict(sd2)                           # same keys, shapes, dtypes

    def test_quantized_roundtrip(self) -> None:
        back = parse_weights(weights_to_text(self.p, quantize=0.02))
        for k in self.p:
            np.testing.assert_allclose(back[k], self.p[k], atol=0.01 + 1e-6)

    def test_messy_and_sparse(self) -> None:
        text = """Sure! Updated weights:
```text
**W1 row 0:** [0.1, −0.2, 0.3, 0.4, 0.5, 0.6]
- W1 row 1  0.1 0.2 0.3
  0.4 0.5 0.6
W1 row 2 is the important one (e.g. 3.2), leave it
W1[3] = 1 2 3 4 5 6
b1: [0, 0, 0, 0, 0, 1e-1]
W2[0,2] = 9
W2[1][0] := -1.5
b2[3] = 2
```"""
        e = parse_weights(text, base=self.p)
        np.testing.assert_allclose(e["W1"][0], [0.1, -0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)
        np.testing.assert_allclose(e["W1"][1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)   # wrapped row
        np.testing.assert_allclose(e["W1"][2], self.p["W1"][2])                            # prose ignored
        np.testing.assert_allclose(e["W1"][3], [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(float(e["b1"][5]), 0.1, places=6)
        self.assertEqual(float(e["W2"][0, 2]), 9.0)
        self.assertEqual(float(e["W2"][1, 0]), -1.5)
        self.assertEqual(float(e["b2"][3]), 2.0)
        self.assertEqual(diff_params(self.p, e)["changed"] > 0, True)
        np.testing.assert_allclose(e["W1"][4:], self.p["W1"][4:])                          # untouched

    def test_strictness(self) -> None:
        with self.assertRaises(ValueError):
            parse_weights("W1 row 0  1 2 3", base=self.p)                       # wrong row length
        with self.assertRaises(ValueError):
            parse_weights("W1 row 0  1 2\nb1  0 0", shapes={"W1": (2, 2), "b1": (2,)})   # missing row 1
        p = parse_weights("W1 row 0  1 2 3", base=self.p, strict=False)        # padded, warns
        np.testing.assert_allclose(p["W1"][0], [1, 2, 3, 0, 0, 0])

    def test_full_without_base(self) -> None:
        p = parse_weights("W1 row 0 1 2\nW1 row 1 3 4\nb1 0.5 -0.5\nW2 row 0 1\nW2 row 1 2\nb2 0")
        self.assertEqual(p["W1"].shape, (2, 2))
        self.assertEqual(p["W2"].shape, (2, 1))
        self.assertEqual(list(p), ["W1", "b1", "W2", "b2"])


class TestLogging(unittest.TestCase):
    def test_result_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = lu.run_dir("expX", lu.new_run_id("t"), td)
            lu.write_result(d, exp_id="expX", tokens_in=10, cost_usd=0.1)
            lu.write_result(d, answer="Breakout", correct=True, cost_usd=None)
            r = json.loads((d / "result.json").read_text())
            self.assertEqual(set(lu.RESULT_FIELDS) - set(r), set())
            self.assertEqual((r["answer"], r["correct"], r["cost_usd"], r["tokens_in"]), ("Breakout", True, 0.1, 10))


class TestBlogData(unittest.TestCase):
    def test_pending_and_done(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import build_blog_data as bbd
        with tempfile.TemporaryDirectory() as td:
            logs, ck, out = Path(td, "logs"), Path(td, "ck"), Path(td, "out")
            bbd.main(["--logs", str(logs), "--checkpoints", str(ck), "--out", str(out), "--exp", "exp1"])
            e = json.loads((out / "exp1.json").read_text())
            self.assertEqual((e["status"], e["runs"], e["table"]["rows"]), ("pending", [], []))
            self.assertEqual(set(e) >= {"exp_id", "title", "status", "summary", "table", "runs", "media"}, True)
            d = lu.run_dir("exp1", "r1", logs)
            (d / "prompt.md").write_text("weights...")
            (d / "response.md").write_text("Game: Breakout")
            lu.write_result(d, exp_id="exp1", run_id="r1", target="Breakout", answer="Breakout", correct=True,
                            cost_usd=0.5)
            (ck / "exp1").mkdir(parents=True)
            (ck / "exp1" / "breakout.json").write_text(json.dumps({"env_id": "ALE/Breakout-v5", "arch": [128, 18]}))
            bbd.main(["--logs", str(logs), "--checkpoints", str(ck), "--out", str(out)])
            e = json.loads((out / "exp1.json").read_text())
            self.assertEqual(e["status"], "done")
            self.assertEqual(e["runs"][0]["answer"], "Breakout")
            self.assertIn("1/1 correct", e["summary"])
            self.assertEqual(e["checkpoints"][0]["env_id"], "ALE/Breakout-v5")
            self.assertEqual(json.loads((out / "exp6.json").read_text())["status"], "pending")


class TestSandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.p = params_from_state_dict(mlp(4, 3, 2).state_dict())
        cls.sb = sb = Sandbox(cls.p, log_dir=cls.tmp.name, timeout=3)

        def forward_pass(x):
            h = np.tanh(np.asarray(x) @ sb.weights["W1"] + sb.weights["b1"])
            return h @ sb.weights["W2"] + sb.weights["b2"]
        sb.register("forward_pass", forward_pass, budget=3, doc="forward_pass(x[B,4]) -> logits[B,2]")
        sb.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sb.close()
        cls.tmp.cleanup()

    def test_persistent_namespace_and_weights(self) -> None:
        self.assertIn("(4, 3)", self.sb.run("W1.shape"))
        self.sb.run("z = 41")
        self.assertIn("42", self.sb.run("print(z + 1)"))

    def test_budget_enforced_and_logged(self) -> None:
        out = self.sb.run("rs = []\nfor i in range(5):\n    try:\n        rs.append(forward_pass(np.ones((1, 4))))\n"
                          "    except BudgetExceeded as e:\n        print('STOP', i, e)\n        break\nlen(rs)")
        self.assertIn("STOP 3", out)
        self.assertEqual(self.sb.budget_status()["forward_pass"], {"used": 3, "budget": 3})
        self.assertIn("BudgetExceeded", self.sb.run("forward_pass(np.ones((1, 4)))"))
        calls = [json.loads(l) for l in (Path(self.tmp.name) / "sandbox_calls.jsonl").read_text().splitlines()]
        self.assertEqual(sum(c["name"] == "forward_pass" and c["ok"] for c in calls), 3)

    def test_no_env_or_processes(self) -> None:
        for code in ("import gymnasium", "import harness.claude_client", "import subprocess",
                     "import os; os.system('true')", "import train"):
            out = self.sb.run(code)
            self.assertTrue("not allowed" in out, f"{code!r} -> {out}")
        self.assertNotIn("ANTHROPIC", self.sb.run("import os; [k for k in os.environ if 'ANTHROPIC' in k]")
                         .replace("[]", ""))

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS-only")
    def test_repo_and_network_unreadable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in (root / "PLAN.md", root / ".env", root / "harness" / "sandbox.py"):
            self.assertIn("PermissionError", self.sb.run(f"open({str(path)!r}).read(1)"), path)
        self.assertIn("PermissionError", self.sb.run(f"__import__('os').listdir({str(root / 'checkpoints')!r})"))
        hf = Path.home() / ".cache"
        self.assertIn("PermissionError", self.sb.run(f"__import__('os').listdir({str(hf)!r})"))
        self.assertIn("ok", self.sb.run("open('scratch.txt', 'w').write('x'); print('ok')"))

    def test_save_weights_paths(self) -> None:
        self.assertIn("saved", self.sb.run("W2b = W2.copy(); W2b[0, 0] = 5.0; save_weights(W2=W2b)"))
        self.assertEqual(float(self.sb.weights["W2"][0, 0]), 5.0)
        self.assertIn("ValueError", self.sb.run("save_weights(W2=np.zeros((9, 9)))"))
        h = self.sb.handlers()
        self.assertIn("saved", h["save_weights"]({"text": "b2[1] = 3.5"}))
        self.assertEqual(float(self.sb.weights["b2"][1]), 3.5)
        self.assertTrue((Path(self.tmp.name) / "weights" / "latest.npz").exists())
        names = [t["name"] for t in self.sb.tools()]
        self.assertEqual(names, ["python", "save_weights"])

    def test_timeout_and_errors(self) -> None:
        self.sb.run("keep = 1")
        self.assertIn("TimeoutError", self.sb.run("while True: pass"))
        self.assertIn("1", self.sb.run("keep"))                         # interrupted, namespace kept
        self.assertIn("ZeroDivisionError", self.sb.run("1/0"))
        big = self.sb.run("print('x' * 100000)")
        self.assertLess(len(big), 21000)


@unittest.skipUnless(has_api_key() and not os.environ.get("SKIP_LIVE"), "no ANTHROPIC_API_KEY")
class TestLiveSmoke(unittest.TestCase):
    def test_tiny_prompt_logs(self) -> None:
        r = ask("Reply with exactly the word: pong", exp_id="_smoke", run_id=lu.new_run_id("tiny"),
                effort="low", max_tokens=2000, condition="smoke")
        self.assertIn("pong", r.text.lower())
        for f in ("prompt.md", "transcript.jsonl", "response.md", "result.json"):
            self.assertTrue((r.log_dir / f).exists(), f)
        res = json.loads((r.log_dir / "result.json").read_text())
        self.assertGreater(res["tokens_in"], 0)
        self.assertGreater(res["cost_usd"], 0)
        print(f"\n[live tiny] tokens_in={res['tokens_in']} tokens_out={res['tokens_out']} cost=${res['cost_usd']:.5f}")

    def test_tool_loop(self) -> None:
        tools = [{"name": "add", "description": "Add two integers.", "input_schema": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}}]
        r = ask("Use the add tool to compute 1234 + 4321, then reply with just the number.", exp_id="_smoke",
                run_id=lu.new_run_id("tools"), effort="low", max_tokens=2000, tools=tools,
                tool_handlers={"add": lambda i: str(i["a"] + i["b"])}, max_turns=3)
        self.assertGreaterEqual(len(r.tool_calls), 1)
        self.assertIn("5555", r.text)
        print(f"\n[live tools] turns={r.n_turns} cost=${r.cost_usd:.5f}")


if __name__ == "__main__":
    unittest.main()
