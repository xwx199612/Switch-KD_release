"""Regression checks for evaluator debug-directory selection and verification."""

import runpy
import tempfile
from pathlib import Path


_MODULE = runpy.run_path("scripts/evaluate_tracker_batch.py")
_effective_debug_dir = _MODULE["_effective_debug_dir"]
_container_debug_output_dir = _MODULE["_container_debug_output_dir"]
_verify_gradient_artifacts = _MODULE["_verify_gradient_artifacts"]
_suffixes = _MODULE["GRADIENT_DEBUG_SUFFIXES"]


def _create_gradient_files(directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in _suffixes:
        (directory / f"{stem}{suffix}").write_bytes(b"diagnostic")


def test_no_debug_dir_uses_output_adjacent_debug_directory():
    with tempfile.TemporaryDirectory() as root:
        output = Path(root) / "output" / "results.jsonl"
        debug_dir = _effective_debug_dir(output, None)
        assert debug_dir == Path(root) / "output" / "debug"
        _create_gradient_files(debug_dir, "frame_a")
        assert _verify_gradient_artifacts(debug_dir, "frame_a") == []
        assert all((debug_dir / f"frame_a{suffix}").is_file() for suffix in _suffixes)


def test_explicit_debug_dir_is_preserved():
    with tempfile.TemporaryDirectory() as root:
        output = Path(root) / "output" / "results.jsonl"
        explicit = Path(root) / "review-artifacts"
        assert _effective_debug_dir(output, explicit) == explicit


def test_docker_mapping_uses_known_output_mount_only():
    with tempfile.TemporaryDirectory() as root:
        output_root = Path(root) / "output"
        host_debug = output_root / "debug"
        assert _container_debug_output_dir(host_debug, output_root, "runtime") == Path("/output/debug")
        outside = Path(root) / "elsewhere"
        assert _container_debug_output_dir(outside, output_root, "runtime") == outside.resolve()


def test_multiple_frames_keep_distinct_gradient_files():
    with tempfile.TemporaryDirectory() as root:
        debug_dir = Path(root) / "output" / "debug"
        _create_gradient_files(debug_dir, "frame_a")
        _create_gradient_files(debug_dir, "frame_b")
        assert _verify_gradient_artifacts(debug_dir, "frame_a") == []
        assert _verify_gradient_artifacts(debug_dir, "frame_b") == []
        for suffix in _suffixes:
            assert (debug_dir / f"frame_a{suffix}").exists()
            assert (debug_dir / f"frame_b{suffix}").exists()
            assert (debug_dir / f"frame_a{suffix}") != (debug_dir / f"frame_b{suffix}")

