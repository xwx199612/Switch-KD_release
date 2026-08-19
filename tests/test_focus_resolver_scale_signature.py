import math

from app.vlm_distill.focus_resolver import FocusResolver


def candidate(index: int, width: float, height: float) -> dict:
    return {
        "index": index,
        "prepared_bbox": [0.0, 0.0, width, height],
    }


def signature(candidate_item: dict, peers: list[dict]) -> dict:
    return FocusResolver._scale_signature_diagnostic(candidate_item, peers)


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
