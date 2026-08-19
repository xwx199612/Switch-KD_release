"""Lightweight checks for FocusResolver's shared gradient debug exporter."""

import os
import tempfile

from PIL import Image

from app.vlm_distill.focus_resolver import FocusResolver


def _evidence(index=0):
    return [{
        "index": index,
        "visual_cell_bbox": [2, 2, 30, 30],
        "recovered_visual_bbox_valid": False,
        "recovered_visual_bbox": None,
        "recovered_visual_semantic_core_bbox": [7, 7, 25, 25],
        "recovered_visual_search_bands": {},
        "recovered_visual_left_candidates": [],
        "recovered_visual_right_candidates": [],
        "recovered_visual_top_candidates": [],
        "recovered_visual_bottom_candidates": [],
    }]


def _load_gray(path):
    return Image.open(path).convert("L")


def test_gradient_debug_exports_shared_maps_and_exact_dimensions():
    image = Image.new("RGB", (32, 32), (20, 20, 20))
    pixels = image.load()
    for y in range(32):
        for x in range(16, 32):
            pixels[x, y] = (220, 220, 220)
    field = FocusResolver._build_global_gradient_field(image)
    before = [row[:] for row in field["vertical_edge_map"]]
    with tempfile.TemporaryDirectory() as output_dir:
        paths = FocusResolver._save_gradient_debug_artifacts(
            image, {0: [8, 8, 24, 24]}, _evidence(), field,
            output_dir=output_dir, frame_stem="frame_a",
        )
        for key in (
            "focus_debug_gradient_luma_path",
            "focus_debug_gradient_color_path",
            "focus_debug_gradient_vertical_path",
            "focus_debug_gradient_horizontal_path",
            "focus_debug_gradient_fused_path",
            "focus_debug_gradient_vertical_recovery_path",
            "focus_debug_gradient_horizontal_recovery_path",
        ):
            assert os.path.exists(paths[key])
            assert Image.open(paths[key]).size == image.size
    assert field["vertical_edge_map"] == before


def test_directional_and_chromatic_evidence_are_visible():
    vertical = Image.new("RGB", (32, 32), (100, 0, 0))
    vp = vertical.load()
    for y in range(32):
        for x in range(16, 32):
            vp[x, y] = (0, 30, 0)
    vfield = FocusResolver._build_global_gradient_field(vertical)
    with tempfile.TemporaryDirectory() as output_dir:
        vpaths = FocusResolver._save_gradient_debug_artifacts(
            vertical, {0: [8, 8, 24, 24]}, _evidence(), vfield,
            output_dir=output_dir, frame_stem="vertical",
        )
        vmap = _load_gray(vpaths["focus_debug_gradient_vertical_path"])
        cmap = _load_gray(vpaths["focus_debug_gradient_color_path"])
    assert vmap.getpixel((16, 16)) > vmap.getpixel((5, 16))
    assert cmap.getpixel((16, 16)) > cmap.getpixel((5, 16))

    horizontal = Image.new("RGB", (32, 32), (20, 20, 20))
    hp = horizontal.load()
    for y in range(16, 32):
        for x in range(32):
            hp[x, y] = (220, 220, 220)
    hfield = FocusResolver._build_global_gradient_field(horizontal)
    with tempfile.TemporaryDirectory() as output_dir:
        hpaths = FocusResolver._save_gradient_debug_artifacts(
            horizontal, {0: [8, 8, 24, 24]}, _evidence(), hfield,
            output_dir=output_dir, frame_stem="horizontal",
        )
        hmap = _load_gray(hpaths["focus_debug_gradient_horizontal_path"])
    assert hmap.getpixel((16, 16)) > hmap.getpixel((16, 5))


def test_invalid_recovery_overlay_is_safe():
    image = Image.new("RGB", (24, 18), (40, 40, 40))
    field = FocusResolver._build_global_gradient_field(image)
    with tempfile.TemporaryDirectory() as output_dir:
        paths = FocusResolver._save_gradient_debug_artifacts(
            image, {0: [4, 3, 20, 15]}, _evidence(), field,
            output_dir=output_dir, frame_stem="invalid",
        )
        assert Image.open(paths["focus_debug_gradient_vertical_recovery_path"]).size == image.size
        assert Image.open(paths["focus_debug_gradient_horizontal_recovery_path"]).size == image.size


def test_gradient_debug_uses_frame_specific_names_without_overwrite():
    image = Image.new("RGB", (12, 12), (20, 20, 20))
    field = FocusResolver._build_global_gradient_field(image)
    with tempfile.TemporaryDirectory() as output_dir:
        first = FocusResolver._save_gradient_debug_artifacts(
            image, {0: [2, 2, 10, 10]}, _evidence(), field,
            output_dir=output_dir, frame_stem="frame_a",
        )
        second = FocusResolver._save_gradient_debug_artifacts(
            image, {0: [2, 2, 10, 10]}, _evidence(), field,
            output_dir=output_dir, frame_stem="frame_b",
        )
        assert first["focus_debug_gradient_luma_path"].endswith("frame_a_gradient_luma.jpg")
        assert second["focus_debug_gradient_luma_path"].endswith("frame_b_gradient_luma.jpg")
        assert first["focus_debug_gradient_luma_path"] != second["focus_debug_gradient_luma_path"]
        assert os.path.exists(first["focus_debug_gradient_luma_path"])
        assert os.path.exists(second["focus_debug_gradient_luma_path"])
