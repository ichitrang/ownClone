import json
import tempfile
import unittest
from pathlib import Path

from training.train_data import convert


class TrainDataTests(unittest.TestCase):
    def test_convert_writes_jsonl(self):
        source = [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ]
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            in_file = Path(tmp) / "in.json"
            out_file = Path(tmp) / "out.jsonl"
            in_file.write_text(json.dumps(source), encoding="utf-8")

            count = convert(in_file, out_file)

            self.assertEqual(count, 1)
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertIn("[INST]", row["text"])
            self.assertIn("hello", row["text"])


if __name__ == "__main__":
    unittest.main()
