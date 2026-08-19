from PIL import Image, ImageDraw

from app.vlm_distill.focus_resolver import FocusResolver


def run_channels(image, boxes, peer_set=None):
    evidence = [
        {
            "index": index,
            "recovered_visual_bbox": list(bbox) if bbox is not None else None,
            "recovered_visual_bbox_valid": bbox is not None,
            "visual_cell_bbox": [0, 0, image.width, image.height],
        }
        for index, bbox in enumerate(boxes)
    ]
    field = FocusResolver._build_global_gradient_field(image)
    FocusResolver._apply_v8_recovered_tri_channel(
        evidence,
        [peer_set or list(range(len(evidence)))],
        image,
        field,
    )
    return evidence


def test_outline_ring_is_peer_unique():
    image = Image.new("RGB", (180, 70), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    boxes = [(10, 10, 50, 50), (70, 10, 110, 50), (130, 10, 170, 50)]
    for bbox in boxes:
        draw.rectangle(bbox, fill=(100, 100, 100))
    draw.rectangle(boxes[1], outline=(250, 250, 250), width=2)
    result = run_channels(image, boxes)
    assert result[1]["recovered_outline_score"] > result[0]["recovered_outline_score"]
    assert result[1]["recovered_outline_ring_continuity"] > 0.0


def test_broad_highlight_beats_neutral_peers():
    image = Image.new("RGB", (180, 70), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    boxes = [(10, 10, 50, 50), (70, 10, 110, 50), (130, 10, 170, 50)]
    for index, bbox in enumerate(boxes):
        draw.rectangle(bbox, fill=(80, 80, 80) if index != 1 else (30, 100, 220))
    result = run_channels(image, boxes)
    assert result[1]["recovered_highlight_score"] > result[0]["recovered_highlight_score"]
    assert result[1]["recovered_highlight_peer_count"] == 2


def test_uniform_recovered_scale_is_positive():
    image = Image.new("RGB", (180, 80), (20, 20, 20))
    boxes = [(10, 10, 50, 50), (70, 10, 110, 50), (125, 5, 175, 55)]
    result = run_channels(image, boxes)
    assert result[2]["recovered_enlargement_available"]
    assert result[2]["recovered_enlargement_score"] > 0.0
    assert result[2]["recovered_scale_isotropy_score"] > 0.9


def test_width_only_recovered_scale_is_suppressed():
    image = Image.new("RGB", (200, 80), (20, 20, 20))
    boxes = [(10, 10, 50, 50), (70, 10, 110, 50), (125, 10, 185, 50)]
    result = run_channels(image, boxes)
    assert result[2]["recovered_enlargement_score"] < 0.2


def test_invalid_recovery_does_not_fallback_to_semantic_geometry():
    image = Image.new("RGB", (120, 60), (40, 40, 40))
    result = run_channels(image, [None, (60, 10, 100, 50)])
    first = result[0]
    assert not first["recovered_geometry_available"]
    assert not first["recovered_outline_available"]
    assert not first["recovered_highlight_available"]
    assert not first["recovered_enlargement_available"]
    assert first["recovered_enlargement_score"] == 0.0


def test_neutral_candidate_has_no_peer_unique_outline_or_highlight():
    image = Image.new("RGB", (180, 70), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    boxes = [(10, 10, 50, 50), (70, 10, 110, 50), (130, 10, 170, 50)]
    for bbox in boxes:
        draw.rectangle(bbox, fill=(80, 80, 80))
    result = run_channels(image, boxes)
    assert max(item["recovered_outline_score"] for item in result) < 0.05
    assert max(item["recovered_highlight_score"] for item in result) < 0.05


def test_multi_label_outline_and_enlargement_can_coexist():
    image = Image.new("RGB", (190, 80), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    boxes = [(10, 15, 50, 55), (70, 15, 110, 55), (125, 5, 185, 65)]
    for index, bbox in enumerate(boxes):
        draw.rectangle(bbox, fill=(100, 100, 100))
        if index == 2:
            draw.rectangle(bbox, outline=(250, 250, 250), width=2)
    result = run_channels(image, boxes)
    assert result[2]["recovered_outline_score"] > 0.0
    assert result[2]["recovered_enlargement_score"] > 0.0
    assert set(result[2]["recovered_focus_signature_scores"]) == {"outline", "highlight", "enlargement"}
