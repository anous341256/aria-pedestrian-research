"""Fast CPU checks for the adapter and unchanged author tracker; no weights/data."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_tracktrack_video.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
API = None


class AdapterTests(unittest.TestCase):
    def test_decode_six_preserves_author_formula_and_dtype(self):
        result = runner.decode_six_columns(torch.tensor([[2, 4, 22, 44, .8, .5, 0]], dtype=torch.float16),
                                           2, 100, 100)
        self.assertEqual(result.dtype, np.float16)
        np.testing.assert_allclose(result[0, :4], [1, 2, 11, 22])
        self.assertAlmostEqual(float(result[0, 4]), .4, places=3)
        self.assertEqual(result[0, 5], 0)

    def test_invalid_rounded_crop_does_not_change_normal_boxes(self):
        dets = np.array([[1.1, 2, 1.2, 8, .9, 0], [-4, 2, 8, 20, .9, 0],
                         [99.7, 2, 100.2, 20, .9, 0], [1, 1, 10, 20, .9, 0]], dtype=np.float32)
        original = dets.copy()
        np.testing.assert_array_equal(runner.crop_valid_mask(dets, 100, 100), [False, True, False, True])
        np.testing.assert_array_equal(dets, original)

    def test_decode_accepts_output_created_in_inference_mode(self):
        with torch.inference_mode():
            output = torch.tensor([[2, 4, 22, 44, .8, .5, 0]], dtype=torch.float16)
        result = runner.decode_six_columns(output, 2, 100, 100)
        self.assertAlmostEqual(float(result[0, 4]), .4, places=3)
        self.assertAlmostEqual(float(output[0, 4]), .8, places=3)

    def test_two_nms_do_not_mutate_shared_raw(self):
        # cxcywh,obj,class. Third box overlaps second enough to distinguish NMS.
        raw = torch.tensor([[[20., 30., 10., 20., .9, .9],
                             [70., 30., 10., 20., .9, .8],
                             [70.5, 30., 10., 20., .8, .8]]])
        before = raw.clone()
        low = API.postprocess(raw.clone(), 1, .1, .80)[0]
        high = API.postprocess(raw.clone(), 1, .1, .95)[0]
        self.assertTrue(torch.equal(raw, before))
        self.assertEqual(len(low), 2)
        self.assertEqual(len(high), 3)

    def test_cmc_validation_rejects_offset_and_nonidentity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmc.txt"
            path.write_text("0\t1\t0\t0\t0\t1\t0\n1\t1\t0\t2\t0\t1\t0\n")
            self.assertEqual(runner.validate_cmc(path)["rows"], 2)
            path.write_text("0\t1\t0\t2\t0\t1\t0\n")
            with self.assertRaises(ValueError):
                runner.validate_cmc(path)
            path.write_text("15\t1\t0\t0\t0\t1\t0\n")
            with self.assertRaises(ValueError):
                runner.validate_cmc(path)

    def test_official_tracker_empty_frame_advances_cmc_and_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "input_cmc.txt"
            path.write_text("".join(f"{i}\t1\t0\t0\t0\t1\t0\n" for i in range(6)))
            args = runner.fixed_tracker_args(30, 200, 200)
            det = np.zeros((1, 2054), dtype=np.float32)
            det[0, :6] = [50, 20, 80, 140, .95, 0]
            det[0, 6] = 1
            empty = np.empty((0, 2054), dtype=np.float32)
            original_cwd = Path.cwd()
            with runner.official_tracker(API, args, output, path) as tracker:
                self.assertEqual(len(runner.step_tracker(tracker, det, det)), 0)
                self.assertEqual(len(runner.step_tracker(tracker, det, det)), 0)
                tracks = runner.step_tracker(tracker, det, det)
                self.assertEqual(len(tracks), 1)
                old_id = tracks[0].track_id
                self.assertEqual(runner.step_tracker(tracker, empty, empty), [])
                self.assertEqual(tracker.frame_id, 4)
                tracks = runner.step_tracker(tracker, det, det)
                self.assertEqual(tracks[0].track_id, old_id)
                self.assertEqual(tracker.frame_id, 5)
                self.assertTrue(tracker.cmc.gmcFile.readline().startswith("5\t"))
            self.assertEqual(Path.cwd(), original_cwd)

    def test_output_filter(self):
        make = lambda box, idx=1: SimpleNamespace(x1y1wh=np.array(box), track_id=idx, score=.9)
        kept, rejected = runner.filtered_tracks([make([0, 0, 10, 20]), make([0, 0, 50, 20]), make([0, 0, 2, 4])])
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, {"aspect_ratio": 1, "area_or_id": 1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args, rest = parser.parse_known_args()
    API = runner.load_official_api(args.repo.resolve())
    unittest.main(argv=[__file__, *rest])
