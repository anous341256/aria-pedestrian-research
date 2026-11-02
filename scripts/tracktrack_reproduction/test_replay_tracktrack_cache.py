"""CPU-only replay contract tests; no detector/ReID weights or video are opened."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import replay_tracktrack_cache as replay
import run_tracktrack_video as runner


REPO = None


def make_source(directory):
    source = directory / "source"
    source.mkdir()
    (source / "detections").mkdir()
    cmc = source / "input_cmc.txt"
    cmc.write_text("".join(f"{i}\t1\t0\t{0 if i == 0 else 2}\t0\t1\t0\n" for i in range(6)))
    args = runner.fixed_tracker_args(10, 200, 200)
    api = replay.load_tracker_api(REPO)
    with runner.official_tracker(api, args, source, cmc) as tracker, (source / "tracks.txt").open("w") as handle:
        for frame in range(1, 7):
            count = 0 if frame == 4 else 1
            data = np.zeros((count, 2054), dtype=np.float16)
            if count:
                data[0, :6] = [50, 20, 80, 140, .95, 0]
                data[0, 6] = 1
            np.savez_compressed(source / "detections" / f"{frame:06d}.npz", frame_id=np.int64(frame),
                                det80=data[:, :6], det95=data[:, :6],
                                crop_valid80=np.ones(count, dtype=bool), crop_valid95=np.ones(count, dtype=bool),
                                tracker_input80=data, tracker_input95=data)
            kept, _ = runner.filtered_tracks(runner.step_tracker(tracker, data, data))
            for track_id, x, y, w, h, score in kept:
                handle.write(f"{frame},{track_id},{round(x, 1)},{round(y, 1)},{round(w, 1)},{round(h, 1)},{round(score, 2)},-1,-1,-1\n")
    runner.write_json(source / "summary.json", {
        "status": "complete", "result": {"frames": 6}, "video_metadata": {"fps": 10, "width": 200, "height": 200},
        "tracker_parameters": vars(args), "seed": runner.SEED, "repository": runner.git_info(REPO),
        "runner": runner.file_info(Path(runner.__file__), True),
    })
    return source, cmc


class ReplayTests(unittest.TestCase):
    def test_same_cmc_is_byte_identical_and_no_gpu_or_models(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            source, cmc = make_source(directory)
            report = replay.replay(SimpleNamespace(run_dir=source, repo=REPO, output=directory / "replay", cmc=cmc, identity_cmc=False))
            self.assertTrue(report["comparison_to_source"]["tracks_byte_identical"])
            self.assertTrue(report["require_byte_identical_tracks"])
            self.assertEqual(report["result"]["frames_without_primary_detections"], 1)
            self.assertEqual(report["result"]["cmc_rows_consumed"], 6)
            self.assertEqual(report["tracker_parameters"]["max_time_lost"], 20)
            self.assertFalse(report["cuda_used"])
            self.assertFalse(report["models_loaded"])
            self.assertEqual(report["cache"]["tracker_input_dtypes"], ["float16"])

    def test_identity_is_sensitivity_only(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            source, _ = make_source(directory)
            report = replay.replay(SimpleNamespace(run_dir=source, repo=REPO, output=directory / "identity", cmc=None, identity_cmc=True))
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["cmc_matrices_equal_to_source_for_processed_frames"])
            self.assertFalse(report["require_byte_identical_tracks"])
            self.assertEqual(report["result"]["frames"], 6)
            self.assertNotIn("accuracy", report["result"])

    def test_incomplete_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = make_source(Path(d))
            value = json.loads((source / "summary.json").read_text())
            value["status"] = "initializing"
            runner.write_json(source / "summary.json", value)
            with self.assertRaisesRegex(ValueError, "must be complete"):
                replay.read_source(source)

    def test_gap_in_cached_frames_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source, _ = make_source(Path(d))
            (source / "detections/000003.npz").rename(source / "detections/misnamed.npz")
            with self.assertRaisesRegex(ValueError, "consecutive"):
                replay.read_source(source)

    def test_feature_row_misalignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "000001.npz"
            det = np.zeros((1, 6), dtype=np.float16)
            inputs = np.ones((1, 2054), dtype=np.float16)
            np.savez(path, frame_id=np.int64(1), det80=det, crop_valid80=np.array([True]), tracker_input80=inputs)
            with self.assertRaisesRegex(ValueError, "alignment"):
                replay.read_cache(path, 1)

    def test_same_cmc_byte_mismatch_is_error_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            source, cmc = make_source(directory)
            with (source / "tracks.txt").open("a") as handle:
                handle.write("6,999,1,1,10,20,0.9,-1,-1,-1\n")
            output = directory / "mismatch"
            with self.assertRaisesRegex(RuntimeError, "byte-identical"):
                replay.replay(SimpleNamespace(run_dir=source, repo=REPO, output=output, cmc=cmc, identity_cmc=False))
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["comparison_to_source"]["tracks_byte_identical"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args, rest = parser.parse_known_args()
    REPO = args.repo.resolve()
    unittest.main(argv=[__file__, *rest])
