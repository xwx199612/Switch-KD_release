from PIL import Image, ImageDraw

from app.vlm_distill.focus_resolver import FocusResolver


def recover(image, bbox, cell=None):
    item = {
        "index": 0,
        "prepared_bbox": list(bbox),
        "visual_cell_bbox": list(cell or (0, 0, image.width, image.height)),
    }
    field = FocusResolver._build_global_gradient_field(image)
    FocusResolver._apply_v7_boundary_recovery([item], image, field)
    return item


def test_clear_rectangle_recovers_enclosing_bbox():
    image = Image.new("RGB", (100, 100), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((25, 25, 75, 75), fill=(220, 220, 220))
    result = recover(image, (30, 30, 70, 70))
    assert result["recovered_visual_bbox_valid"]
    assert result["recovered_visual_left_delta"] >= 3
    assert result["recovered_visual_right_delta"] >= 3
    assert result["recovered_boundary_side_count"] == 4


def test_semantic_bbox_can_recover_inward_to_true_rectangle():
    image = Image.new("RGB", (100, 100), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((25, 25, 75, 75), fill=(220, 220, 220))
    result = recover(image, (20, 20, 80, 80))
    assert result["recovered_visual_bbox_valid"]
    assert result["recovered_visual_bbox"][0] >= 22
    assert result["recovered_visual_bbox"][2] <= 78


def test_internal_rectangle_is_rejected_when_it_cuts_through_semantic_core():
    image = Image.new("RGB", (160, 120), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 145, 105), fill=(100, 100, 100))
    draw.rectangle((50, 35, 110, 85), fill=(230, 230, 230))
    result = recover(image, (35, 25, 125, 95), (0, 0, 160, 120))
    assert result["recovered_visual_bbox_reason"] in {
        "semantic_core_not_enclosed",
        "no_valid_joint_boundary",
        "insufficient_side_candidates",
    }
    if result["recovered_visual_bbox_valid"]:
        assert result["recovered_visual_semantic_core_contained"]


def test_v71_reports_core_and_enclosure_diagnostics():
    image = Image.new("RGB", (100, 100), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((25, 25, 75, 75), fill=(220, 220, 220))
    result = recover(image, (30, 30, 70, 70))
    assert result["focus_boundary_recovery_version"] == "v7.1-enclosing-boundary-diagnostic"
    assert result["recovered_visual_semantic_core_bbox"] is not None
    assert 0.0 <= result["recovered_visual_enclosure_support"] <= 1.0


def test_color_only_boundary_uses_color_gradient():
    # Approximate equal luminance, with a strong chromatic transition.
    image = Image.new("RGB", (100, 100), (150, 0, 0))
    ImageDraw.Draw(image).rectangle((25, 25, 75, 75), fill=(0, 45, 0))
    result = recover(image, (30, 30, 70, 70))
    assert result["recovered_visual_bbox_valid"]


def test_fragmented_internal_lines_do_not_replace_card_boundary():
    image = Image.new("RGB", (120, 100), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 20, 95, 80), fill=(210, 210, 210))
    for x in range(40, 86, 8):
        draw.line((x, 40, x + 3, 45), fill=(10, 10, 10), width=1)
    result = recover(image, (35, 30, 85, 70))
    if result["recovered_visual_bbox_valid"]:
        assert result["recovered_visual_left_delta"] >= 5
        assert result["recovered_visual_right_delta"] >= 5
    else:
        assert result["recovered_visual_bbox"] is None


def test_neighbor_visual_cell_prevents_boundary_theft():
    image = Image.new("RGB", (140, 80), (15, 15, 15))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 55, 60), fill=(190, 190, 190))
    draw.rectangle((75, 20, 125, 60), fill=(230, 230, 230))
    result = recover(image, (18, 25, 48, 55), (0, 0, 65, 80))
    if result["recovered_visual_bbox_valid"]:
        assert result["recovered_visual_bbox"][2] <= 65
    else:
        assert result["recovered_visual_bbox"] is None


def test_noisy_empty_image_is_safe():
    image = Image.new("RGB", (80, 80), (100, 100, 100))
    result = recover(image, (25, 25, 55, 55))
    assert not result["recovered_visual_bbox_valid"]
    assert result["recovered_visual_bbox"] is None


def test_image_edge_is_boundary_limited_or_conservative():
    image = Image.new("RGB", (80, 80), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((0, 20, 45, 60), fill=(220, 220, 220))
    result = recover(image, (5, 25, 35, 55))
    assert result["recovered_visual_bbox_reason"] in {
        "joint_boundary_recovered",
        "insufficient_side_candidates",
        "no_valid_joint_boundary",
    }
