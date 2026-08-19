from app.vlm_distill.focus_resolver import FocusResolver


def _candidate(coordinate, adjusted, raw=None, enclosure=0.8, distance=None):
    return {
        "coordinate": float(coordinate),
        "adjusted_score": float(adjusted),
        "score": float(raw if raw is not None else adjusted),
        "edge_score": float(raw if raw is not None else adjusted),
        "enclosure_score": float(enclosure),
        "distance_from_semantic_side": float(distance if distance is not None else coordinate - 60),
    }


def test_adjacent_coordinates_collapse_into_spatial_ridge_clusters():
    candidates = [
        _candidate(40, 0.90), _candidate(41, 0.88),
        _candidate(55, 0.80), _candidate(56, 0.79),
        _candidate(80, 0.70),
    ]
    selected, clusters, radius, reserved = FocusResolver._cluster_v7_side_candidates(
        candidates, semantic_coordinate=60, reference_dimension=200,
        outward_sign=1, top_k=4,
    )
    assert radius == 5
    assert len(clusters) == 3
    assert [cluster["member_coordinates"] for cluster in clusters] == [[40.0, 41.0], [55.0, 56.0], [80.0]]
    assert len(selected) == 3
    assert not reserved


def test_weaker_valid_outward_cluster_is_reserved():
    candidates = [
        _candidate(40, 0.95), _candidate(41, 0.94),
        _candidate(55, 0.90), _candidate(56, 0.89),
        _candidate(80, 0.60),
    ]
    selected, _, _, reserved = FocusResolver._cluster_v7_side_candidates(
        candidates, semantic_coordinate=60, reference_dimension=200,
        outward_sign=1, top_k=2,
    )
    assert reserved
    assert len(selected) == 2
    assert any(candidate["coordinate"] == 80.0 for candidate in selected)
    assert any(candidate["retained_by"] == "outward_reservation" for candidate in selected)


def test_no_outward_evidence_does_not_create_a_candidate():
    candidates = [_candidate(40, 0.9), _candidate(41, 0.88)]
    selected, clusters, _, reserved = FocusResolver._cluster_v7_side_candidates(
        candidates, semantic_coordinate=60, reference_dimension=200,
        outward_sign=1, top_k=4,
    )
    assert [candidate["coordinate"] for candidate in selected] == [40.0]
    assert len(clusters) == 1
    assert not reserved


def test_cluster_radius_is_adaptive_and_bounded():
    small = FocusResolver._cluster_v7_side_candidates(
        [_candidate(10, 0.8), _candidate(14, 0.7)],
        semantic_coordinate=20, reference_dimension=20, outward_sign=1, top_k=4,
    )
    large = FocusResolver._cluster_v7_side_candidates(
        [_candidate(10, 0.8), _candidate(19, 0.7)],
        semantic_coordinate=20, reference_dimension=1000, outward_sign=1, top_k=4,
    )
    assert small[2] == 3
    assert large[2] == 8
    assert len(small[1]) == 2
    assert len(large[1]) == 2
