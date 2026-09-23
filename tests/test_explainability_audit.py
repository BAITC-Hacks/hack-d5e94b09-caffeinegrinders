import unittest
from pathlib import Path

from audit_explainability import audit_text


ROOT = Path(__file__).resolve().parents[1]


class ExplainabilityAuditTest(unittest.TestCase):
    def test_audit_contains_evidence_and_breakdown(self):
        text = audit_text(ROOT / "out", sample=2)
        self.assertIn("Explainability audit", text)
        self.assertIn("evidence=", text)
        self.assertIn("breakdown:", text)
        self.assertIn("attention=", text)


if __name__ == "__main__":
    unittest.main()
