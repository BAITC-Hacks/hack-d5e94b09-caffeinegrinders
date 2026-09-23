import unittest
from pathlib import Path

from demo import build_demo


ROOT = Path(__file__).resolve().parents[1]


class DemoScriptTest(unittest.TestCase):
    def test_demo_script_mentions_core_segments(self):
        text = build_demo(ROOT / "out")
        self.assertIn("Демо-сценарий", text)
        self.assertIn("Топ-узел", text)
        self.assertIn("Маршрут пересылки", text)
        self.assertIn("Следующий запрос данных", text)


if __name__ == "__main__":
    unittest.main()
