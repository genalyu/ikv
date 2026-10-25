"""CPU-only regressions for transient shared-filesystem snapshot reads."""
import errno
import unittest
from unittest.mock import patch
from training.parallel_scorer.wandb_bridge import read_pipeline_snapshot

class SnapshotTests(unittest.TestCase):
    def test_transient_missing_and_partial_json(self):
        with patch("pathlib.Path.read_text", side_effect=[FileNotFoundError(errno.ENOENT, "transient"), "{", '{"pipeline": {"status": "running"}}']), patch("training.parallel_scorer.wandb_bridge.time.sleep"):
            self.assertEqual(read_pipeline_snapshot("snapshot")["pipeline"]["status"], "running")

    def test_persistent_missing_is_bounded(self):
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError(errno.ENOENT, "missing")) as read, patch("training.parallel_scorer.wandb_bridge.time.sleep"):
            with self.assertRaises(FileNotFoundError):
                read_pipeline_snapshot("snapshot")
            self.assertEqual(read.call_count, 5)

    def test_permission_error_is_not_hidden(self):
        with patch("pathlib.Path.read_text", side_effect=PermissionError(errno.EACCES, "denied")) as read:
            with self.assertRaises(PermissionError):
                read_pipeline_snapshot("snapshot")
            self.assertEqual(read.call_count, 1)
