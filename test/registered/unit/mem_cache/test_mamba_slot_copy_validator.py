import json
import unittest
from types import SimpleNamespace

from sglang.srt.debug_utils.mamba_slot_copy_validator import (
    LOG_PREFIX,
    MambaSlotCopyValidator,
    SlotSelection,
    build_copy_evidence,
)


class _FakeTensor:
    def __init__(self, values):
        self.values = values if isinstance(values, list) else [values]

    def __getitem__(self, index):
        return _FakeTensor(self.values[index])

    def item(self):
        return self.values[0]

    def detach(self):
        return self

    def cpu(self):
        return self

    def flatten(self):
        return self

    def tolist(self):
        return self.values


class TestMambaSlotCopyValidator(unittest.TestCase):
    def test_matches_selection_to_actual_staged_d2h_source(self):
        validator = MambaSlotCopyValidator(enabled=True)
        req = SimpleNamespace(
            rid="request-a",
            seqlen=640,
            kv_committed_len=640,
            mamba_ping_pong_track_buffer=_FakeTensor([11, 10]),
        )
        batch = SimpleNamespace(
            forward_iter=100,
            seq_lens_cpu=_FakeTensor([639]),
        )

        validator.record_result(req, batch, 0)
        validator.record_selection(
            req=req,
            cache_len=640,
            selected_logical_slot=0,
            selected_slot_tensor=_FakeTensor(11),
        )
        with self.assertLogs(
            "sglang.srt.debug_utils.mamba_slot_copy_validator", level="ERROR"
        ) as logs:
            validator.record_staged_d2h(
                _FakeTensor([11]),
                io_backend="kernel",
                layout="page_first",
            )

        evidence = json.loads(logs.output[0].split(f"{LOG_PREFIX} ", 1)[1])
        self.assertEqual(evidence["status"], "WRONG_SLOT_COPIED")
        self.assertEqual(evidence["selected_slot_id"], 11)
        self.assertEqual(evidence["d2h_src_slot_id"], 11)

    def test_proves_staged_d2h_copied_future_checkpoint_slot(self):
        evidence = build_copy_evidence(
            SlotSelection(
                rid="request-a",
                result_forward_iter=100,
                result_seq_len=639,
                request_seq_len=640,
                kv_committed_len=640,
                cache_len=640,
                selected_logical_slot=0,
                selected_slot_id=11,
                ping_pong_slot_ids=(11, 10),
            ),
            d2h_src_slot_id=11,
            io_backend="kernel",
            layout="page_first",
        )

        self.assertEqual(evidence["status"], "WRONG_SLOT_COPIED")
        self.assertTrue(evidence["copied_selected_slot"])
        self.assertTrue(evidence["selected_future_checkpoint"])
        self.assertTrue(evidence["wrong_slot_copied"])

    def test_does_not_report_when_d2h_copies_another_slot(self):
        evidence = build_copy_evidence(
            SlotSelection(
                rid="request-a",
                result_forward_iter=100,
                result_seq_len=639,
                request_seq_len=640,
                kv_committed_len=640,
                cache_len=640,
                selected_logical_slot=0,
                selected_slot_id=11,
                ping_pong_slot_ids=(11, 10),
            ),
            d2h_src_slot_id=10,
            io_backend="kernel",
            layout="page_first",
        )

        self.assertFalse(evidence["copied_selected_slot"])
        self.assertFalse(evidence["wrong_slot_copied"])


if __name__ == "__main__":
    unittest.main()
