from app.vlm_distill.focus_resolver import FocusResolver


def arbitrate(items, peer_sets):
    FocusResolver._apply_v8_1_group_first_arbitration(items, peer_sets)
    return items


def candidate(index, outline=0.0, highlight=0.0, enlargement=0.0):
    return {
        "index": index,
        "recovered_outline_score": outline,
        "recovered_highlight_score": highlight,
        "recovered_enlargement_score": enlargement,
        "recovered_outline_available": True,
        "recovered_highlight_available": True,
        "recovered_enlargement_available": True,
    }


def test_group_enlargement_beats_isolated_outline():
    items = [
        candidate(0, enlargement=0.10),
        candidate(1, enlargement=0.80),
        candidate(2, enlargement=0.12),
        candidate(3, outline=0.95),
    ]
    arbitrate(items, [[0, 1, 2]])
    assert items[1]["recovered_focus_arbitration_source"] == "peer_group"
    assert items[1]["recovered_focus_arbitration_candidate_index"] == 1
    assert items[3]["recovered_isolated_focus_arbitration"]["reason"] == "blocked_by_grouped_focus"


def test_group_highlight_beats_isolated_outline():
    items = [candidate(0, highlight=0.08), candidate(1, highlight=0.82), candidate(2, highlight=0.10), candidate(3, outline=0.95)]
    arbitrate(items, [[0, 1, 2]])
    assert items[1]["recovered_focus_arbitration_candidate_index"] == 1
    assert items[1]["recovered_focus_arbitration_channels"] == ["highlight"]


def test_group_outline_beats_isolated_highlight():
    items = [candidate(0, outline=0.10), candidate(1, outline=0.84), candidate(2, outline=0.12), candidate(3, highlight=0.95)]
    arbitrate(items, [[0, 1, 2]])
    assert items[1]["recovered_focus_arbitration_candidate_index"] == 1


def test_isolated_fallback_runs_without_group_winner():
    items = [candidate(0), candidate(1), candidate(2, outline=0.90)]
    arbitrate(items, [[0, 1]])
    assert items[2]["recovered_focus_arbitration_source"] == "isolated_fallback"
    assert items[2]["recovered_focus_arbitration_candidate_index"] == 2


def test_ambiguous_isolated_candidates_do_not_force_winner():
    items = [candidate(0, outline=0.80), candidate(1, highlight=0.81)]
    arbitrate(items, [])
    assert items[0]["recovered_focus_arbitration_source"] == "none"
    assert items[0]["recovered_focus_arbitration_candidate_index"] is None


def test_multiple_groups_compare_group_winners():
    items = [
        candidate(0, enlargement=0.70), candidate(1, enlargement=0.10),
        candidate(2, highlight=0.85), candidate(3, highlight=0.10),
    ]
    arbitrate(items, [[0, 1], [2, 3]])
    assert items[2]["recovered_focus_arbitration_candidate_index"] == 2
    assert items[2]["recovered_focus_arbitration_source"] == "peer_group"


def test_pure_enlargement_can_win_without_other_channels():
    items = [candidate(0, enlargement=0.05), candidate(1, enlargement=0.78), candidate(2, enlargement=0.06)]
    arbitrate(items, [[0, 1, 2]])
    assert items[1]["recovered_focus_arbitration_candidate_index"] == 1
    assert items[1]["recovered_focus_arbitration_channels"] == ["enlargement"]


def test_same_candidate_can_receive_multiple_channel_votes():
    items = [candidate(0, outline=0.75, enlargement=0.72), candidate(1, outline=0.05, enlargement=0.05), candidate(2, outline=0.06, enlargement=0.06)]
    arbitrate(items, [[0, 1, 2]])
    assert set(items[0]["recovered_focus_arbitration_channels"]) == {"outline", "enlargement"}
