"""Process-level regressions for the CLI benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("benchmark.py")
SPEC = importlib.util.spec_from_file_location("supergrep_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkProcessTest(unittest.TestCase):
    def make_fake_cli(self, directory: Path) -> Path:
        fake = directory / "fake_cli.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "assert sys.argv[-2:] == ['--batch-size', '4']\n"
            "print(json.dumps({'model': {'id': 'compact-multilingual'}, "
            "'mode': 'fast', 'stats': {'timing': {"
            "'model_load_ms': 1, 'tokenization_ms': 1, 'inference_ms': 1, "
            "'discovery_ms': 1, 'chunking_ms': 1, 'search_ms': 1, 'total_ms': 1}, "
            "'chunks_total': 1, "
            "'chunks_evaluated': 1, 'scan_complete': True, 'partial': False, "
            "'scoring_complete': True}, 'query_seen': sys.argv[1], "
            "'query': sys.argv[1], 'large_output': 'x' * 8388608}))\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake

    def test_large_stdout_drains_and_distinct_queries_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fake = self.make_fake_cli(directory)
            corpus = directory / "corpus"
            corpus.mkdir()
            (corpus / "a.rs").write_text("fn a() {}\n", encoding="utf-8")
            queries = directory / "queries.jsonl"
            query_records = [
                {"id": "q1", "split": "development", "query_type": "relevant", "query": "where is retry?"},
                {"id": "q2", "split": "development", "query_type": "relevant", "query": "어떤 설정이 적용되나?"},
            ]
            queries.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in query_records) + "\n",
                encoding="utf-8",
            )
            output = directory / "result.json"
            argv = [
                str(MODULE_PATH), str(fake), str(corpus), "--runs", "2",
                "--queries-file", str(queries), "--output", str(output),
                "--timeout-seconds", "5", "--batch-size", "4",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(benchmark.main(), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["query_set"]["distinct_queries"], 2)
            self.assertEqual([run["query_id"] for run in result["runs"]], ["q1", "q2"])
            self.assertEqual([run["query"] for run in result["runs"]], ["where is retry?", "어떤 설정이 적용되나?"])
            self.assertTrue(all(run["wall_ms"] > 0 for run in result["runs"]))
            self.assertEqual(result["batch_size"], 4)


if __name__ == "__main__":
    unittest.main()
