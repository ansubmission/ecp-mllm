from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import _bootstrap  # noqa: F401

from ecp_mllm.config import PathsConfig
from ecp_mllm.data.cfc_adapter import CFCAdapter
from ecp_mllm.eval.counting import count_tracks_like_cfc, read_mot_tracks


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "cfc"


def _write_png(path: Path, width: int, height: int) -> None:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    row = b"\x00" + (b"\x00\x00\x00" * width)
    raw = row * height
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, level=9)),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


class CFCCountingTests(unittest.TestCase):
    def test_count_tracks_like_cfc_matches_expected_crossings(self) -> None:
        tracks = read_mot_tracks(FIXTURE_ROOT / "annotations" / "kenai-val" / "clip_a" / "gt.txt")
        counts = count_tracks_like_cfc(tracks, width=100, height=50, filter_dist=0.05)
        self.assertEqual(counts.left, 1)
        self.assertEqual(counts.right, 1)

    def test_adapter_loads_clip_records_and_ground_truth_counts(self) -> None:
        adapter = CFCAdapter(
            PathsConfig(
                cfc_metadata_path=FIXTURE_ROOT / "metadata",
                cfc_annotations_path=FIXTURE_ROOT / "annotations",
                cfc_raw_root=FIXTURE_ROOT / "raw",
                cfc_3channel_root=FIXTURE_ROOT / "3channel",
            )
        )
        clips = adapter.load_clip_records(["kenai-val"])
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].clip_id, "clip_a")
        labels = adapter.derive_ground_truth_counts(["kenai-val"])
        self.assertEqual(labels[0].counts.left, 1)
        self.assertEqual(labels[0].counts.right, 1)

    def test_adapter_reconstructs_clips_from_file_lists_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "file_lists").mkdir()
            (root / "images" / "kenai").mkdir(parents=True)
            (root / "mot" / "kenai-val" / "flat_clip").mkdir(parents=True)
            for index in range(10):
                _write_png(root / "images" / "kenai" / f"flat_clip_{index}.png", width=100, height=50)
            (root / "file_lists" / "kenai-val.txt").write_text(
                "\n".join(
                    f"data/cfc_v1.1/images/kenai/flat_clip_{index}.png"
                    for index in range(10)
                ),
                encoding="utf-8",
            )
            gt_source = FIXTURE_ROOT / "annotations" / "kenai-val" / "clip_a" / "gt.txt"
            (root / "mot" / "kenai-val" / "flat_clip" / "gt.txt").write_text(
                gt_source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            adapter = CFCAdapter(
                PathsConfig(
                    cfc_data_path=root,
                    cfc_annotations_path=root / "mot",
                    cfc_raw_root=root / "images",
                )
            )
            clips = adapter.load_clip_records(["kenai-val"])
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0].clip_id, "flat_clip")
            self.assertEqual(clips[0].width, 100)
            self.assertEqual(clips[0].height, 50)
            self.assertEqual(len(clips[0].metadata["frame_paths_by_variant"]["raw"]), 10)
            labels = adapter.derive_ground_truth_counts(["kenai-val"])
            self.assertEqual(labels[0].counts.left, 1)
            self.assertEqual(labels[0].counts.right, 1)


if __name__ == "__main__":
    unittest.main()

