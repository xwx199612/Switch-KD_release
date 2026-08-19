"""Regression checks for evaluator debug-directory selection and verification."""

import runpy
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


_MODULE = runpy.run_path("scripts/evaluate_tracker_batch.py")
_effective_debug_dir = _MODULE["_effective_debug_dir"]
_effective_debug_dir_for_run = _MODULE["_effective_debug_dir_for_run"]
_container_gradient_staging_dir = _MODULE["_container_gradient_staging_dir"]
_save_peer_debug_image = _MODULE["save_peer_debug_image"]
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


def test_docker_default_debug_dir_is_under_repository_output_mount():
    output = Path("/repo/v8_gradient_map/results.jsonl")
    selected = _effective_debug_dir_for_run(output, None)
    assert selected == Path("/repo/v8_gradient_map/debug")


def test_docker_uses_internal_staging_not_user_visible_directory():
    staging = _container_gradient_staging_dir("run123", "frame_a")
    assert staging == Path("/output/.focus_gradient_staging/run123/frame_a")


def test_explicit_debug_dir_is_used_even_with_docker_staging():
    with tempfile.TemporaryDirectory() as root:
        output = Path(root) / "foo" / "results.jsonl"
        explicit = Path(root) / "custom_debug"
        assert _effective_debug_dir_for_run(output, explicit) == explicit.resolve()


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


def test_gradient_artifacts_are_collected_to_host_and_metadata_paths_translated():
    fields = (
        ("focus_debug_gradient_luma_path", "_gradient_luma.jpg"),
        ("focus_debug_gradient_color_path", "_gradient_color.jpg"),
        ("focus_debug_gradient_vertical_path", "_gradient_vertical.jpg"),
        ("focus_debug_gradient_horizontal_path", "_gradient_horizontal.jpg"),
        ("focus_debug_gradient_fused_path", "_gradient_fused.jpg"),
        ("focus_debug_gradient_vertical_recovery_path", "_gradient_vertical_recovery_debug.jpg"),
        ("focus_debug_gradient_horizontal_recovery_path", "_gradient_horizontal_recovery_debug.jpg"),
    )
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        staging = root / "staging"
        host = root / "v8_gradient_map" / "debug"
        staging.mkdir()
        host.mkdir(parents=True)
        sources = {}
        for field, suffix in fields:
            source = staging / f"frame_a{suffix}"
            source.write_bytes(b"gradient")
            sources[field] = str(source)
        metadata_source = staging / "frame_a_cv_prepared.json"
        metadata_source.write_text(json.dumps({"gradient": sources}), encoding="utf-8")
        response = {"tracker_debug": {"focus_resolver_debug": {
            **sources,
            "focus_cv_prepared_metadata_path": str(metadata_source),
        }}}
        _save_peer_debug_image(response, "frame_a.png", str(host), None)
        for _, suffix in fields:
            assert (host / f"frame_a{suffix}").is_file()
        translated = json.loads((host / "frame_a_cv_prepared.json").read_text(encoding="utf-8"))
        assert all(str(host / f"frame_a{suffix}") in translated["gradient"].values() for _, suffix in fields)


def test_docker_staging_artifacts_are_collected_to_host():
    fields = (
        ("focus_debug_gradient_luma_path", "_gradient_luma.jpg"),
        ("focus_debug_gradient_color_path", "_gradient_color.jpg"),
        ("focus_debug_gradient_vertical_path", "_gradient_vertical.jpg"),
        ("focus_debug_gradient_horizontal_path", "_gradient_horizontal.jpg"),
        ("focus_debug_gradient_fused_path", "_gradient_fused.jpg"),
        ("focus_debug_gradient_vertical_recovery_path", "_gradient_vertical_recovery_debug.jpg"),
        ("focus_debug_gradient_horizontal_recovery_path", "_gradient_horizontal_recovery_debug.jpg"),
    )
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        host = root / "v8_gradient_map" / "debug"
        host.mkdir(parents=True)
        sources = {field: f"/output/.focus_gradient_staging/run/frame_a{suffix}" for field, suffix in fields}
        payloads = {source: b"gradient" for source in sources.values()}

        def fake_run(command, **kwargs):
            source = command[-1]
            return SimpleNamespace(stdout=payloads.get(source, b"{}"))

        response = {"tracker_debug": {"focus_resolver_debug": dict(sources)}}
        with patch.object(_MODULE["subprocess"], "run", side_effect=fake_run):
            _save_peer_debug_image(response, "frame_a.png", str(host), "runtime")
        assert all((host / f"frame_a{suffix}").is_file() for _, suffix in fields)
