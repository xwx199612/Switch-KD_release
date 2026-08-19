import math

from app.vlm_distill.focus_resolver import FocusResolver


def candidate(index: int, width: float, height: float) -> dict:
    return {
        "index": index,
        "prepared_bbox": [0.0, 0.0, width, height],
    }


def signature(candidate_item: dict, peers: list[dict]) -> dict:
    return FocusResolver._scale_signature_diagnostic(candidate_item, peers)


def dual_signature(candidate_item: dict, peers: list[dict]) -> dict:
    return FocusResolver._scale_signature_dual_diagnostic(candidate_item, peers)


def with_visual(item: dict, width: float, height: float, source: str = "recovered") -> dict:
    result = dict(item)
    result["recovered_current_container_bbox"] = [0.0, 0.0, width, height]
    result["recovered_current_container_valid"] = source == "recovered"
    if source == "extent":
        result["visual_extent_bbox"] = [0.0, 0.0, width, height]
        result["extent_valid"] = True
    return result


def test_uniform_enlargement_has_positive_isotropic_signature():
    first = candidate(0, 100, 100)
    second = candidate(1, 100, 100)
    focused = candidate(2, 120, 120)

    result = signature(focused, [first, second, focused])

    assert result["scale_positive_magnitude"] > 0.0
    assert result["scale_isotropy_score"] > 0.9
    assert result["scale_aspect_preservation_score"] > 0.9
    assert result["scale_measure_agreement_score"] > 0.9
    assert result["scale_signature_score"] > 0.2


def test_one_axis_stretch_is_lower_than_uniform_enlargement():
    peers = [candidate(0, 100, 100), candidate(1, 100, 100)]
    uniform = signature(candidate(2, 120, 120), peers + [candidate(2, 120, 120)])
    width_only = signature(candidate(2, 130, 100), peers + [candidate(2, 130, 100)])
    height_only = signature(candidate(2, 100, 130), peers + [candidate(2, 100, 130)])

    assert width_only["scale_signature_score"] < uniform["scale_signature_score"]
    assert height_only["scale_signature_score"] < uniform["scale_signature_score"]
    assert width_only["scale_isotropy_score"] < uniform["scale_isotropy_score"]
    assert height_only["scale_isotropy_score"] < uniform["scale_isotropy_score"]


def test_aspect_change_is_suppressed():
    peers = [candidate(0, 100, 100), candidate(1, 100, 100)]
    result = signature(candidate(2, 140, 80), peers + [candidate(2, 140, 80)])

    assert result["scale_aspect_preservation_score"] < 0.2
    assert result["scale_isotropy_score"] < 0.2
    assert result["scale_signature_score"] < 0.1


def test_smaller_candidate_has_no_positive_scale_signature():
    peers = [candidate(0, 100, 100), candidate(1, 100, 100)]
    result = signature(candidate(2, 90, 90), peers + [candidate(2, 90, 90)])

    assert result["scale_positive_magnitude"] == 0.0
    assert result["scale_signature_score"] == 0.0


def test_heterogeneous_peers_reduce_reliability():
    homogeneous = [candidate(0, 100, 100), candidate(1, 100, 100)]
    heterogeneous = [candidate(0, 80, 100), candidate(1, 120, 100)]
    focused = candidate(2, 130, 110)

    homogeneous_result = signature(focused, homogeneous + [focused])
    heterogeneous_result = signature(focused, heterogeneous + [focused])

    assert heterogeneous_result["scale_peer_reliability"] < homogeneous_result["scale_peer_reliability"]
    assert heterogeneous_result["scale_signature_score"] < homogeneous_result["scale_signature_score"]


def test_singleton_and_degenerate_geometry_are_safe():
    singleton = candidate(0, 100, 100)
    singleton_result = signature(singleton, [singleton])
    degenerate = signature(
        {"index": 0, "prepared_bbox": [0.0, 0.0, 0.0, 100.0]},
        [candidate(1, 100, 100)],
    )

    assert singleton_result["scale_peer_count"] == 0
    assert singleton_result["scale_signature_score"] == 0.0
    assert degenerate["scale_signature_score"] == 0.0
    for result in (singleton_result, degenerate):
        for key, value in result.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(value)


def test_v61_semantic_values_match_v60_values():
    peers = [candidate(0, 100, 100), candidate(1, 100, 100)]
    focused = candidate(2, 120, 120)
    old = signature(focused, peers + [focused])
    dual = dual_signature(focused, peers + [focused])
    for key in old:
        if key.startswith("scale_"):
            assert dual[key] == old[key]


def test_visual_space_detects_growth_missing_from_semantic_space():
    peers = [with_visual(candidate(0, 100, 100), 100, 100), with_visual(candidate(1, 100, 100), 100, 100)]
    focused = with_visual(candidate(2, 100, 100), 120, 120)
    result = dual_signature(focused, peers + [focused])
    assert result["semantic_scale_signature_score"] == 0.0
    assert result["visual_scale_signature_score"] > 0.2
    assert result["scale_space_signature_delta"] > 0.0
    assert result["scale_space_relation"] == "visual_stronger"
    assert result["visual_scale_geometry_source"] == "recovered_current_container_bbox"


def test_semantic_only_growth_is_distinguished_from_visual_space():
    peers = [with_visual(candidate(0, 100, 100), 100, 100), with_visual(candidate(1, 100, 100), 100, 100)]
    focused = with_visual(candidate(2, 120, 120), 100, 100)
    result = dual_signature(focused, peers + [focused])
    assert result["semantic_scale_signature_score"] > 0.2
    assert result["visual_scale_signature_score"] == 0.0
    assert result["scale_space_relation"] == "semantic_stronger"


def test_visual_width_only_stretch_is_suppressed():
    peers = [with_visual(candidate(0, 100, 100), 100, 100), with_visual(candidate(1, 100, 100), 100, 100)]
    uniform = dual_signature(with_visual(candidate(2, 100, 100), 120, 120), peers + [with_visual(candidate(2, 100, 100), 120, 120)])
    stretch = dual_signature(with_visual(candidate(2, 100, 100), 130, 100), peers + [with_visual(candidate(2, 100, 100), 130, 100)])
    assert stretch["visual_scale_signature_score"] < uniform["visual_scale_signature_score"]


def test_invalid_or_missing_visual_geometry_is_safe():
    peers = [candidate(0, 100, 100), candidate(1, 100, 100)]
    focused = candidate(2, 120, 120)
    result = dual_signature(focused, peers + [focused])
    assert result["visual_scale_geometry_source"] == "unavailable"
    assert result["visual_scale_peer_count"] == 0
    assert result["visual_scale_signature_score"] == 0.0
    assert result["scale_space_relation"] == "semantic_only"

    invalid = with_visual(focused, 0, 100)
    invalid_result = dual_signature(invalid, peers + [invalid])
    assert invalid_result["visual_scale_signature_score"] == 0.0
    for key, value in invalid_result.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)


def test_visual_baseline_accepts_deterministic_mixed_sources():
    first = with_visual(candidate(0, 100, 100), 100, 100)
    second = with_visual(candidate(1, 100, 100), 100, 100, source="extent")
    focused = with_visual(candidate(2, 100, 100), 120, 120)
    result = dual_signature(focused, [first, second, focused])
    assert result["visual_scale_peer_count"] == 2
    assert result["visual_scale_peer_median_width"] == 100.0
    assert result["visual_scale_geometry_source"] == "recovered_current_container_bbox"
