"""Dedicated single-image navigation-focus resolution."""

from __future__ import annotations

import json
import math
import tempfile
import time
from typing import Any

from PIL import Image, ImageDraw
from PIL import ImageFont

from .state_tracker import StateObservationError


FOCUS_RESOLVER_PROMPT_TEMPLATE = """Inspect the screenshot and choose the candidate with visible navigation focus.

Candidate numbers are drawn directly on the image. Compare the annotated
candidate containers in their shared visual context; the number in the image
exactly matches the index in the Candidates list.
When shown as separated group crops, each crop preserves the original local
layout of multiple candidates from the same screenshot. Candidate numbers mark
their corresponding elements inside each crop. The original navigation-focus
appearance is preserved; compare border, outline, scale, ring, elevation, and
emphasis.
The image may be arranged into separated UI groups cropped from the same
original screenshot; candidate indices remain the original indices.

Candidates are already detected UI elements and have been restricted to
geometrically comparable peer groups.
Do not detect new elements.

Candidate annotations identify parsed UI elements. Their boxes may cover the
element itself or only part of a larger visual container. Use the surrounding
visual context in the image to determine whether the candidate belongs to a
focused card, button, icon, tab, or other container. Inspect the annotated
elements and their surrounding context directly. Within each visually
comparable peer group, compare:

1. full container width and height,
2. scale relative to neighboring peers,
3. visible outer border or outline,
4. focus ring,
5. elevation or container emphasis.

A visible outer focus ring, bright or contrasting outline, highlighted
container background, visibly enlarged focused container, or clearly unique
elevation/glow is strong direct focus evidence. Direct visible focus evidence
must dominate semantic salience. A size difference can be sufficient when it
clearly distinguishes one peer.

Semantic relevance, text importance, content brightness by itself, a colorful
icon by itself, being a heading or title, and being first in a row are weak or
non-focus evidence and must not determine focus.

Choose the visually distinguished candidate, not the most semantically salient
candidate.

Return null only when no candidate is visually distinguishable.

Candidates:
{candidates}

Return only:
{"focused_index": integer or null}"""


class FocusResolver:
    """Resolve visual navigation focus with an already-loaded VLM engine."""

    MAX_NEW_TOKENS = 64
    ROI_MARGIN_FRACTION = 0.08
    ROI_MAX_AREA_FRACTION = 0.75
    ENLARGE_FACTOR = 2.0
    MAX_ENLARGED_DIMENSION = 2048
    PEER_MIN_HEIGHT_RATIO = 0.5
    PEER_MIN_VERTICAL_OVERLAP_RATIO = 0.4
    PEER_MAX_CENTER_Y_DISTANCE_RATIO = 0.5
    PEER_MIN_CENTER_X_DISTANCE_RATIO = 0.5
    PEER_MAX_HORIZONTAL_GAP_RATIO = 2.0
    PEER_MIN_HORIZONTAL_OVERLAP_RATIO = 0.4
    PEER_MAX_VERTICAL_GAP_RATIO = 2.0
    GRID_MAX_VERTICAL_SPAN_HEIGHTS = 5.0
    GRID_MAX_HORIZONTAL_SPAN_WIDTHS = 8.0
    DEBUG_IMAGE_PATH = f"{tempfile.gettempdir()}/focus_resolver_input.png"
    DEBUG_UNANNOTATED_IMAGE_PATH = f"{tempfile.gettempdir()}/focus_resolver_unannotated_input.png"
    DEBUG_PEER_IMAGE_PATH = f"{tempfile.gettempdir()}/focus_resolver_peers.jpg"
    VISUAL_RING_CONTINUITY_WEIGHT = 0.55
    VISUAL_RING_CONTRAST_WEIGHT = 0.30
    VISUAL_BACKGROUND_WEIGHT = 0.15
    SCALE_BALANCE_SIGMA = 0.20
    OUTLINE_EXCLUSIVITY_FLOOR = 0.5
    PEER_MIN_WIDTH_RATIO = 0.6
    PEER_MAX_WIDTH_RATIO = 1.7
    PEER_MIN_HEIGHT_RATIO = 0.6
    PEER_MAX_HEIGHT_RATIO = 1.7
    PEER_MAX_ASPECT_LOG_DELTA = 0.65
    PEER_MIN_AXIS_OVERLAP = 0.30
    PEER_CENTER_ALIGNMENT_FACTOR = 0.75
    MIN_COMPARABLE_PEERS = 1
    OUTLINE_V5_MIN_SCORE = 0.50
    OUTLINE_V5_MIN_MARGIN = 0.15
    ENLARGEMENT_V5_MIN_SCORE = 0.50
    ENLARGEMENT_V5_MIN_MARGIN = 0.15
    ENLARGEMENT_FULL_SCALE_GROWTH = 0.25
    ENLARGEMENT_BALANCE_FLOOR = 0.5
    ENLARGEMENT_MIN_MEANINGFUL_GROWTH = 0.10
    VISUAL_FOOTPRINT_EXPANSIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    VISUAL_FOOTPRINT_SCORE_TOLERANCE = 0.05
    ENLARGEMENT_SIBLING_MIN_WIDTH_RATIO = 0.80
    ENLARGEMENT_SIBLING_MAX_WIDTH_RATIO = 1.25
    ENLARGEMENT_SIBLING_MIN_HEIGHT_RATIO = 0.80
    ENLARGEMENT_SIBLING_MAX_HEIGHT_RATIO = 1.25
    ENLARGEMENT_SIBLING_MAX_ASPECT_LOG_DELTA = 0.20
    ENLARGEMENT_SIBLING_MIN_AXIS_OVERLAP = 0.45
    ENLARGEMENT_V5_MIN_CROSS_SET_MARGIN = 0.10
    ENLARGEMENT_LOCAL_MARGIN_X = 0.50
    ENLARGEMENT_LOCAL_MARGIN_Y = 0.50
    ENLARGEMENT_SYMMETRY_FLOOR = 0.5
    HIGHLIGHT_V5_MIN_SCORE = 0.60
    HIGHLIGHT_V5_MIN_MARGIN = 0.18
    CONTAINER_PROPOSAL_EXPANSIONS = (0.0, 0.05, 0.10, 0.20)

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def resolve(self, image: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(candidates, list):
            raise StateObservationError("focus resolver candidates must be a list")

        indices_before_filter = list(range(len(candidates)))
        candidate_groups = self._focus_candidate_groups(candidates)
        candidate_group_types = self._focus_candidate_group_types(candidates, candidate_groups)
        selected_group_indices = list(range(len(candidate_groups)))
        focus_indices = [
            index
            for group in candidate_groups
            for index in self._spatial_order_indices(candidates, group)
        ]
        group_by_index = {
            index: group_index
            for group_index, group in enumerate(candidate_groups)
            for index in group
        }
        focus_group_ids = [group_by_index[index] for index in focus_indices]
        # Grouping is descriptive and controls presentation order; it must not
        # remove candidates from the resolver input.
        filter_used = False
        filtered_indices = list(focus_indices)
        spatial_order_indices = list(focus_indices)
        focus_candidates = [candidates[index] for index in focus_indices]

        (
            annotated_image,
            unannotated_image,
            roi_bbox,
            roi_used,
            annotated_indices,
            focus_image_mode,
            montage_tile_indices,
            montage_grid,
            montage_size,
            montage_tile_sizes,
            prepared_candidate_bboxes,
        ) = self._prepare_focus_image(
            image,
            focus_candidates,
            focus_indices,
            focus_group_ids=focus_group_ids,
            use_montage=len(candidate_groups) >= 2,
        )
        peer_analysis = self._build_peer_analysis(
            candidate_groups, prepared_candidate_bboxes
        )
        visual_evidence = self._visual_focus_evidence(
            unannotated_image,
            candidates,
            candidate_groups,
            prepared_candidate_bboxes,
            comparison_groups=peer_analysis["peer_sets"],
            peer_group_by_index=peer_analysis["peer_group_by_index"],
        )
        enlargement_sibling_analysis = self._apply_v5_peer_evidence(
            visual_evidence,
            peer_analysis["peer_sets"],
            unannotated_image,
        )
        v5_cascade = self._run_visual_focus_cascade(
            visual_evidence,
            peer_analysis["peer_sets"],
            peer_analysis["peer_group_by_index"],
            peer_analysis["isolated_indices"],
            enlargement_peer_sets=enlargement_sibling_analysis["sibling_sets"],
        )
        focus_debug_image_path: str | None = None
        try:
            annotated_image.save(self.DEBUG_IMAGE_PATH, format="PNG")
            focus_debug_image_path = self.DEBUG_IMAGE_PATH
        except (OSError, ValueError):
            pass
        focus_peer_debug_image_path: str | None = None
        try:
            self._save_peer_debug_image(
                unannotated_image,
                prepared_candidate_bboxes,
                peer_analysis,
                enlargement_sibling_analysis,
            )
            focus_peer_debug_image_path = self.DEBUG_PEER_IMAGE_PATH
        except (OSError, ValueError):
            pass
        focus_debug_unannotated_image_path: str | None = None
        try:
            unannotated_image.save(self.DEBUG_UNANNOTATED_IMAGE_PATH, format="PNG")
            focus_debug_unannotated_image_path = self.DEBUG_UNANNOTATED_IMAGE_PATH
        except (OSError, ValueError):
            pass
        candidate_lines = "\n".join(
            f'{index}. text="{candidate.get("text", "")}" '
            f'bbox={candidate.get("bbox_norm")} '
            f'size=[{candidate["bbox_norm"][2] - candidate["bbox_norm"][0]},'
            f'{candidate["bbox_norm"][3] - candidate["bbox_norm"][1]}]'
            for index, candidate in zip(focus_indices, focus_candidates)
        )
        prompt = FOCUS_RESOLVER_PROMPT_TEMPLATE.replace("{candidates}", candidate_lines)
        started = time.perf_counter()
        raw_output = self.engine.generate_raw(annotated_image, prompt, self.MAX_NEW_TOKENS)
        elapsed = round(time.perf_counter() - started, 3)
        focused_index = self._parse_output(raw_output, len(candidates))
        debug = dict(getattr(self.engine, "last_debug", {}) or {})
        debug.update({
            "mode": "focus_resolution",
            "elapsed_seconds": elapsed,
            "generation_kwargs": {"do_sample": False, "max_new_tokens": self.MAX_NEW_TOKENS},
            "focus_roi_bbox_pixels": list(roi_bbox) if roi_bbox is not None else None,
            "focus_roi_used": roi_used,
            "focus_roi_size": [roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]]
            if roi_bbox is not None else [image.width, image.height],
            "focus_input_size": [annotated_image.width, annotated_image.height],
            "annotated_candidate_indices": annotated_indices,
            "focus_candidate_indices_before_filter": indices_before_filter,
            "focus_candidate_indices_after_filter": filtered_indices,
            "focus_candidate_indices_spatial_order": spatial_order_indices,
            "focus_candidate_filter_used": filter_used,
            "focus_candidate_groups": candidate_groups,
            "focus_candidate_selected_group_indices": selected_group_indices,
            "focus_candidate_group_types": candidate_group_types,
            "focus_group_montage_used": focus_image_mode == "group_montage",
            "focus_group_montage_group_indices": candidate_groups
            if focus_image_mode == "group_montage" else [],
            "focus_annotation_mode": "index_labels_only",
            "focus_image_mode": focus_image_mode,
            "focus_montage_tile_indices": montage_tile_indices,
            "focus_montage_grid": montage_grid,
            "focus_montage_size": montage_size,
            "focus_montage_tile_sizes": montage_tile_sizes,
            "focus_montage_group_tile_count": len(candidate_groups)
            if focus_image_mode == "group_montage" else 0,
            "focus_montage_group_tile_indices": candidate_groups
            if focus_image_mode == "group_montage" else [],
            "focus_debug_image_path": focus_debug_image_path,
            "focus_debug_unannotated_image_path": focus_debug_unannotated_image_path,
            "focus_visual_evidence_space": "prepared",
            "focus_peer_debug_image_path": focus_peer_debug_image_path,
            "focus_peer_groups": peer_analysis["peer_sets"],
            "focus_isolated_indices": peer_analysis["isolated_indices"],
            "focus_peer_debug": peer_analysis["debug"],
            "focus_enlargement_sibling_groups": enlargement_sibling_analysis["sibling_sets"],
            **v5_cascade,
            "focus_visual_evidence": visual_evidence,
            "focus_visual_evidence_top_indices": [
                item["index"] for item in sorted(
                    visual_evidence,
                    key=lambda item: (-item["direct_visual_confidence"], item["index"]),
                )
            ],
        })
        return {
            "focused_index": focused_index,
            "raw_output": raw_output,
            "inference_debug": debug,
        }

    @classmethod
    def _build_peer_analysis(
        cls,
        geometric_groups: list[list[int]],
        prepared_candidate_bboxes: dict[int, list[int]],
    ) -> dict[str, Any]:
        """Build symmetric, geometry-only comparable peer sets."""

        def bbox(index: int) -> tuple[float, float, float, float] | None:
            value = prepared_candidate_bboxes.get(index)
            if not isinstance(value, (list, tuple)) or len(value) < 4:
                return None
            try:
                left, top, right, bottom = (float(value[i]) for i in range(4))
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in (left, top, right, bottom)):
                return None
            if right <= left or bottom <= top:
                return None
            return left, top, right, bottom

        def pair_reason(first: int, second: int) -> str | None:
            first_box = bbox(first)
            second_box = bbox(second)
            if first_box is None or second_box is None:
                return "invalid_geometry"
            fl, ft, fr, fb = first_box
            sl, st, sr, sb = second_box
            fw, fh = fr - fl, fb - ft
            sw, sh = sr - sl, sb - st
            width_ratio = fw / max(sw, 1e-6)
            height_ratio = fh / max(sh, 1e-6)
            if not (
                cls.PEER_MIN_WIDTH_RATIO <= width_ratio <= cls.PEER_MAX_WIDTH_RATIO
                and cls.PEER_MIN_WIDTH_RATIO <= 1.0 / max(width_ratio, 1e-6) <= cls.PEER_MAX_WIDTH_RATIO
            ):
                return "width_ratio_mismatch"
            if not (
                cls.PEER_MIN_HEIGHT_RATIO <= height_ratio <= cls.PEER_MAX_HEIGHT_RATIO
                and cls.PEER_MIN_HEIGHT_RATIO <= 1.0 / max(height_ratio, 1e-6) <= cls.PEER_MAX_HEIGHT_RATIO
            ):
                return "height_ratio_mismatch"
            first_aspect = fw / max(fh, 1e-6)
            second_aspect = sw / max(sh, 1e-6)
            if abs(math.log(max(first_aspect, 1e-6) / max(second_aspect, 1e-6))) > cls.PEER_MAX_ASPECT_LOG_DELTA:
                return "aspect_ratio_mismatch"

            vertical_overlap = max(0.0, min(fb, sb) - max(ft, st)) / max(min(fh, sh), 1e-6)
            horizontal_overlap = max(0.0, min(fr, sr) - max(fl, sl)) / max(min(fw, sw), 1e-6)
            center_y_delta = abs((ft + fb) / 2.0 - (st + sb) / 2.0)
            center_x_delta = abs((fl + fr) / 2.0 - (sl + sr) / 2.0)
            same_row = vertical_overlap >= cls.PEER_MIN_AXIS_OVERLAP or center_y_delta <= cls.PEER_CENTER_ALIGNMENT_FACTOR * max(fh, sh)
            same_column = horizontal_overlap >= cls.PEER_MIN_AXIS_OVERLAP or center_x_delta <= cls.PEER_CENTER_ALIGNMENT_FACTOR * max(fw, sw)
            if not (same_row or same_column):
                return "row_column_mismatch"
            return None

        peer_sets: list[list[int]] = []
        peer_group_by_index: dict[int, int] = {}
        geometric_group_by_index: dict[int, int] = {}
        rejection_reasons: dict[int, dict[str, str]] = {}
        isolated_indices: list[int] = []

        for geometric_group_id, raw_group in enumerate(geometric_groups):
            indices = sorted({int(index) for index in raw_group})
            for index in indices:
                geometric_group_by_index[index] = geometric_group_id
                rejection_reasons.setdefault(index, {})
            parent = {index: index for index in indices}

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(first: int, second: int) -> None:
                first_root, second_root = find(first), find(second)
                if first_root != second_root:
                    parent[second_root] = first_root

            for position, first in enumerate(indices):
                for second in indices[position + 1:]:
                    reason = pair_reason(first, second)
                    if reason is None:
                        union(first, second)
                    else:
                        rejection_reasons[first][str(second)] = reason
                        rejection_reasons[second][str(first)] = reason

            components: dict[int, list[int]] = {}
            for index in indices:
                components.setdefault(find(index), []).append(index)
            for component in sorted((sorted(value) for value in components.values()), key=lambda value: value[0] if value else -1):
                if len(component) >= 2:
                    peer_id = len(peer_sets)
                    peer_sets.append(component)
                    for index in component:
                        peer_group_by_index[index] = peer_id
                else:
                    isolated_indices.extend(component)

        for index in sorted(prepared_candidate_bboxes):
            if index not in peer_group_by_index:
                isolated_indices.append(index)
                geometric_group_by_index.setdefault(index, -1)
                rejection_reasons.setdefault(index, {})

        isolated_indices = sorted(set(isolated_indices))
        debug = []
        for index in sorted(geometric_group_by_index):
            peer_id = peer_group_by_index.get(index, -1)
            comparable = [value for value in (peer_sets[peer_id] if peer_id >= 0 else []) if value != index]
            debug.append({
                "index": index,
                "geometric_group_id": geometric_group_by_index[index],
                "peer_group_id": peer_id,
                "comparable_peer_indices": comparable,
                "peer_count": len(comparable),
                "is_isolated": not comparable,
                "peer_rejection_reasons": rejection_reasons.get(index, {}),
            })
        return {
            "peer_sets": peer_sets,
            "peer_group_by_index": peer_group_by_index,
            "isolated_indices": isolated_indices,
            "debug": debug,
        }

    @classmethod
    def _apply_v5_peer_evidence(
        cls,
        evidence: list[dict[str, Any]],
        peer_sets: list[list[int]],
        image: Image.Image,
    ) -> dict[str, Any]:
        """Rebase diagnostic channel comparisons on V5 comparable peers."""
        by_index = {int(item["index"]): item for item in evidence}
        peer_by_index = {
            index: peer_id
            for peer_id, peer_set in enumerate(peer_sets)
            for index in peer_set
        }
        for item in evidence:
            index = int(item["index"])
            peer_id = peer_by_index.get(index, -1)
            item["peer_group_id"] = peer_id
            item["comparable_peer_indices"] = [
                peer for peer in (peer_sets[peer_id] if peer_id >= 0 else []) if peer != index
            ]
            item["peer_count"] = len(item["comparable_peer_indices"])
            item["is_isolated"] = peer_id < 0

        def median(values: list[float]) -> float:
            ordered = sorted(values)
            if not ordered:
                return 0.0
            middle = len(ordered) // 2
            return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0

        def exclusivity(score: float, peers: list[float]) -> float:
            if not peers:
                return 0.0
            baseline = median(peers)
            return cls._clamp01((score - baseline) / max(0.15, 1.0 - baseline))

        pixels = image.convert("RGB")
        pixel_data = pixels.load()
        image_width, image_height = pixels.size

        def clipped_box(box: Any) -> tuple[float, float, float, float] | None:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                return None
            try:
                left, top, right, bottom = [float(box[index]) for index in range(4)]
            except (TypeError, ValueError):
                return None
            left = max(0.0, min(left, float(image_width)))
            top = max(0.0, min(top, float(image_height)))
            right = max(0.0, min(right, float(image_width)))
            bottom = max(0.0, min(bottom, float(image_height)))
            return (left, top, right, bottom) if right > left and bottom > top else None

        def color_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
            return min(1.0, sum(abs(first[channel] - second[channel]) for channel in range(3)) / 765.0)

        def footprint_score(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            left, top, right, bottom = box
            width = max(2.0, right - left)
            height = max(2.0, bottom - top)
            radius_x = max(1, min(8, int(round(width * 0.015))))
            radius_y = max(1, min(8, int(round(height * 0.015))))
            side_values: list[float] = []
            continuity_values: list[float] = []
            for side in ("top", "bottom", "left", "right"):
                samples = max(5, min(32, int(round((width if side in ("top", "bottom") else height) / 18.0))))
                contrasts: list[float] = []
                for sample_index in range(samples):
                    fraction = (sample_index + 0.5) / samples
                    if side in ("top", "bottom"):
                        x = int(max(0, min(image_width - 1, round(left + fraction * width))))
                        boundary = int(round(top if side == "top" else bottom))
                        inside_y = max(0, min(image_height - 1, boundary + (radius_y if side == "top" else -radius_y)))
                        outside_y = max(0, min(image_height - 1, boundary - radius_y if side == "top" else boundary + radius_y))
                        contrasts.append(color_distance(pixel_data[x, inside_y], pixel_data[x, outside_y]))
                    else:
                        y = int(max(0, min(image_height - 1, round(top + fraction * height))))
                        boundary = int(round(left if side == "left" else right))
                        inside_x = max(0, min(image_width - 1, boundary + (radius_x if side == "left" else -radius_x)))
                        outside_x = max(0, min(image_width - 1, boundary - radius_x if side == "left" else boundary + radius_x))
                        contrasts.append(color_distance(pixel_data[inside_x, y], pixel_data[outside_x, y]))
                side_values.append(sum(contrasts) / max(len(contrasts), 1))
                continuity_values.append(sum(1.0 for value in contrasts if value >= 0.045) / max(len(contrasts), 1))
            boundary_contrast = sum(side_values) / 4.0
            boundary_coherence = sum(1.0 for value in side_values if value >= 0.035) / 4.0
            edge_continuity = sum(continuity_values) / 4.0

            inner_samples: list[float] = []
            for row in range(4):
                for column in range(4):
                    x = int(max(0, min(image_width - 1, round(left + (column + 0.5) * width / 4.0))))
                    y = int(max(0, min(image_height - 1, round(top + (row + 0.5) * height / 4.0))))
                    color = pixel_data[x, y]
                    inner_samples.append(sum(color) / 765.0)
            interior_mean = sum(inner_samples) / max(len(inner_samples), 1)
            interior_variance = sum((value - interior_mean) ** 2 for value in inner_samples) / max(len(inner_samples), 1)
            interior_coherence = cls._clamp01(1.0 - math.sqrt(interior_variance) * 4.0)
            score = cls._clamp01(
                0.45 * boundary_coherence
                + 0.30 * boundary_contrast
                + 0.15 * edge_continuity
                + 0.10 * interior_coherence
            )
            return score, boundary_coherence, boundary_contrast, edge_continuity

        def detect_visual_footprint(item: dict[str, Any]) -> None:
            semantic = clipped_box(item.get("prepared_bbox"))
            cell = clipped_box(item.get("visual_cell_bbox")) or semantic
            if semantic is None or cell is None:
                return
            left, top, right, bottom = semantic
            width = right - left
            height = bottom - top
            proposals: list[dict[str, Any]] = []
            for expansion in cls.VISUAL_FOOTPRINT_EXPANSIONS:
                proposal = clipped_box((
                    max(cell[0], left - width * expansion),
                    max(cell[1], top - height * expansion),
                    min(cell[2], right + width * expansion),
                    min(cell[3], bottom + height * expansion),
                ))
                if proposal is None:
                    continue
                score, boundary, contrast, continuity = footprint_score(proposal)
                proposals.append({
                    "bbox": [round(value, 2) for value in proposal],
                    "expansion_ratio": expansion,
                    "score": score,
                    "boundary": boundary,
                    "contrast": contrast,
                    "continuity": continuity,
                })
            if not proposals:
                selected = semantic
                selected_score = 0.0
                source = "semantic_bbox_fallback"
                selected_expansion = 0.0
            else:
                best_score = max(proposal["score"] for proposal in proposals)
                eligible = [proposal for proposal in proposals if proposal["score"] >= best_score - cls.VISUAL_FOOTPRINT_SCORE_TOLERANCE]
                selected_proposal = min(eligible, key=lambda proposal: proposal["expansion_ratio"])
                selected = tuple(float(value) for value in selected_proposal["bbox"])
                selected_score = selected_proposal["score"]
                source = "visual_container"
                selected_expansion = selected_proposal["expansion_ratio"]
            item["visual_footprint_bbox"] = [round(value, 2) for value in selected]
            item["visual_footprint_width"] = max(0.0, selected[2] - selected[0])
            item["visual_footprint_height"] = max(0.0, selected[3] - selected[1])
            item["visual_footprint_area"] = item["visual_footprint_width"] * item["visual_footprint_height"]
            item["visual_footprint_expansion_ratio"] = selected_expansion
            item["visual_footprint_score"] = selected_score
            item["visual_footprint_source"] = source
            item["visual_footprint_proposals"] = proposals[:3]

        def measure_direct_extent(item: dict[str, Any]) -> None:
            semantic = clipped_box(item.get("prepared_bbox"))
            cell = clipped_box(item.get("visual_cell_bbox"))
            if semantic is None or cell is None:
                item["enlargement_extent_reason"] = "invalid_geometry"
                return
            left, top, right, bottom = semantic
            width = right - left
            height = bottom - top
            observation = clipped_box((
                max(cell[0], left - width * cls.ENLARGEMENT_LOCAL_MARGIN_X),
                max(cell[1], top - height * cls.ENLARGEMENT_LOCAL_MARGIN_Y),
                min(cell[2], right + width * cls.ENLARGEMENT_LOCAL_MARGIN_X),
                min(cell[3], bottom + height * cls.ENLARGEMENT_LOCAL_MARGIN_Y),
            )) or semantic

            def luminance(color: tuple[int, int, int]) -> float:
                return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255.0

            def sample_band(side: str, distance: int, samples: int) -> tuple[float, float]:
                values: list[float] = []
                gradients: list[float] = []
                for sample_index in range(samples):
                    fraction = (sample_index + 0.5) / samples
                    if side in ("left", "right"):
                        y = int(max(0, min(image_height - 1, round(top + fraction * height))))
                        x = int(round(left - distance if side == "left" else right + distance))
                        x = max(int(observation[0]), min(int(observation[2]) - 1, x))
                        inside_x = max(int(observation[0]), min(int(observation[2]) - 1, int(round(left + (2 if side == "left" else -2)))))
                        outside = luminance(pixel_data[x, y])
                        inside = luminance(pixel_data[inside_x, y])
                    else:
                        x = int(max(0, min(image_width - 1, round(left + fraction * width))))
                        y = int(round(top - distance if side == "top" else bottom + distance))
                        y = max(int(observation[1]), min(int(observation[3]) - 1, y))
                        inside_y = max(int(observation[1]), min(int(observation[3]) - 1, int(round(top + (2 if side == "top" else -2)))))
                        outside = luminance(pixel_data[x, y])
                        inside = luminance(pixel_data[x, inside_y])
                    values.append(outside)
                    gradients.append(abs(outside - inside))
                return sum(values) / max(len(values), 1), sum(gradients) / max(len(gradients), 1)

            reference_values: list[float] = []
            for row in range(3):
                for column in range(3):
                    x = int(max(0, min(image_width - 1, round(left + (column + 0.5) * width / 3.0))))
                    y = int(max(0, min(image_height - 1, round(top + (row + 0.5) * height / 3.0))))
                    reference_values.append(luminance(pixel_data[x, y]))
            reference = sum(reference_values) / max(len(reference_values), 1)
            reference_variance = sum((value - reference) ** 2 for value in reference_values) / max(len(reference_values), 1)
            step = max(1, min(4, int(round(min(width, height) * 0.025))))
            growth: dict[str, int] = {"left": 0, "right": 0, "top": 0, "bottom": 0}
            side_support: dict[str, float] = {}
            samples = max(5, min(15, int(round(max(width, height) / 20.0))))
            for side in growth:
                maximum = int(round((left - observation[0]) if side == "left" else (observation[2] - right) if side == "right" else (top - observation[1]) if side == "top" else (observation[3] - bottom)))
                consecutive = 0
                best = 0
                support_values: list[float] = []
                for distance in range(step, maximum + 1, step):
                    outside_mean, gradient = sample_band(side, distance, samples)
                    continuity = cls._clamp01(1.0 - abs(outside_mean - reference) / max(0.12, math.sqrt(reference_variance) * 3.0 + 0.04))
                    edge_support = cls._clamp01(gradient / 0.10)
                    support = 0.70 * continuity + 0.30 * edge_support
                    support_values.append(support)
                    if support >= 0.48:
                        consecutive += 1
                        best = distance
                    elif consecutive >= 2:
                        break
                    else:
                        consecutive = 0
                growth[side] = best
                side_support[side] = sum(support_values[-2:]) / max(1, len(support_values[-2:])) if support_values else 0.0

            extent = clipped_box((left - growth["left"], top - growth["top"], right + growth["right"], bottom + growth["bottom"])) or semantic
            horizontal_balance = min(growth["left"], growth["right"]) / max(growth["left"], growth["right"], 1.0)
            vertical_balance = min(growth["top"], growth["bottom"]) / max(growth["top"], growth["bottom"], 1.0)
            item.update({
                "enlargement_local_observation_bbox": [round(value, 2) for value in observation],
                "visual_extent_bbox": [round(value, 2) for value in extent],
                "visual_extent_width": max(0.0, extent[2] - extent[0]),
                "visual_extent_height": max(0.0, extent[3] - extent[1]),
                "visual_extent_area": max(0.0, extent[2] - extent[0]) * max(0.0, extent[3] - extent[1]),
                "visual_extent_left_growth": growth["left"],
                "visual_extent_right_growth": growth["right"],
                "visual_extent_top_growth": growth["top"],
                "visual_extent_bottom_growth": growth["bottom"],
                "visual_extent_left_growth_ratio": growth["left"] / max(width, 1e-6),
                "visual_extent_right_growth_ratio": growth["right"] / max(width, 1e-6),
                "visual_extent_top_growth_ratio": growth["top"] / max(height, 1e-6),
                "visual_extent_bottom_growth_ratio": growth["bottom"] / max(height, 1e-6),
                "extent_horizontal_balance": horizontal_balance,
                "extent_vertical_balance": vertical_balance,
                "extent_symmetry": math.sqrt(max(0.0, horizontal_balance * vertical_balance)),
                "enlargement_extent_reason": "direct_visual_extent",
                "extent_side_support": side_support,
            })

        for item in evidence:
            detect_visual_footprint(item)
            measure_direct_extent(item)

        for peer_set in peer_sets:
            available = [by_index[index] for index in peer_set if index in by_index]
            for item in available:
                index = int(item["index"])
                others = [candidate for candidate in available if int(candidate["index"]) != index]
                absolute = float(item.get("absolute_outline_score", item.get("raw_outline_score", 0.0)))
                outline_excl = exclusivity(absolute, [float(candidate.get("absolute_outline_score", candidate.get("raw_outline_score", 0.0))) for candidate in others])
                item["absolute_outline_score"] = absolute
                item["raw_outline_score"] = absolute
                item["outline_exclusivity"] = outline_excl
                item["outline_exclusivity_gate"] = cls.OUTLINE_EXCLUSIVITY_FLOOR + (1.0 - cls.OUTLINE_EXCLUSIVITY_FLOOR) * outline_excl
                item["outline_score"] = cls._clamp01(absolute * item["outline_exclusivity_gate"])

            boxes = [candidate.get("visual_footprint_bbox") for candidate in available]
            if not all(isinstance(box, (list, tuple)) and len(box) >= 4 for box in boxes):
                continue
            widths = [max(1.0, float(box[2]) - float(box[0])) for box in boxes]
            heights = [max(1.0, float(box[3]) - float(box[1])) for box in boxes]
            areas = [width * height for width, height in zip(widths, heights)]
            base_width, base_height, base_area = median(widths), median(heights), median(areas)
            semantic_boxes = [candidate.get("prepared_bbox") for candidate in available]
            semantic_widths = [max(1.0, float(box[2]) - float(box[0])) if isinstance(box, (list, tuple)) and len(box) >= 4 else 1.0 for box in semantic_boxes]
            semantic_heights = [max(1.0, float(box[3]) - float(box[1])) if isinstance(box, (list, tuple)) and len(box) >= 4 else 1.0 for box in semantic_boxes]
            semantic_areas = [width * height for width, height in zip(semantic_widths, semantic_heights)]
            semantic_base_width = median(semantic_widths)
            semantic_base_height = median(semantic_heights)
            semantic_base_area = median(semantic_areas)
            for position, (item, width, height, area) in enumerate(zip(available, widths, heights, areas)):
                semantic_width = semantic_widths[position]
                semantic_height = semantic_heights[position]
                semantic_area = semantic_areas[position]
                item["relative_width"] = semantic_width / max(semantic_base_width, 1e-6)
                item["relative_height"] = semantic_height / max(semantic_base_height, 1e-6)
                item["relative_area"] = semantic_area / max(semantic_base_area, 1e-6)
                relative_visual_width = width / max(base_width, 1e-6)
                relative_visual_height = height / max(base_height, 1e-6)
                relative_visual_area = area / max(base_area, 1e-6)
                relative_width = relative_visual_width
                relative_height = relative_visual_height
                relative_area = relative_visual_area
                width_growth = max(0.0, relative_width - 1.0)
                height_growth = max(0.0, relative_height - 1.0)
                uniform_growth = min(width_growth, height_growth)
                ratio = max(relative_width, 1e-6) / max(relative_height, 1e-6)
                scale_balance = math.exp(-abs(math.log(max(ratio, 1e-6))) / cls.SCALE_BALANCE_SIGMA)
                uniqueness = exclusivity(relative_area, [areas[index] / max(base_area, 1e-6) for index in range(len(areas)) if index != position])
                consistency = float(item.get("peer_size_consistency", 1.0))
                protrusion = float(item.get("peer_protrusion_score", 0.0))
                uniform_scale = math.sqrt(max(relative_width * relative_height, 0.0))
                scale_growth = max(0.0, uniform_scale - 1.0)
                base_enlargement_score = cls._clamp01(
                    scale_growth / max(cls.ENLARGEMENT_FULL_SCALE_GROWTH, 1e-6)
                )
                scale_balance_gate = (
                    cls.ENLARGEMENT_BALANCE_FLOOR
                    + (1.0 - cls.ENLARGEMENT_BALANCE_FLOOR) * scale_balance
                )
                two_axis_support = cls._clamp01(
                    uniform_growth / max(cls.ENLARGEMENT_MIN_MEANINGFUL_GROWTH, 1e-6)
                )
                scale_evidence = cls._clamp01(
                    base_enlargement_score * scale_balance_gate * two_axis_support
                )
                item.update({
                    "relative_visual_width": relative_visual_width,
                    "relative_visual_height": relative_visual_height,
                    "relative_visual_area": relative_visual_area,
                    "width_growth": width_growth,
                    "height_growth": height_growth,
                    "uniform_growth": uniform_growth,
                    "scale_balance": scale_balance,
                    "uniform_scale": uniform_scale,
                    "scale_growth": scale_growth,
                    "scale_balance_gate": scale_balance_gate,
                    "two_axis_support": two_axis_support,
                    "base_enlargement_score": base_enlargement_score,
                    "scale_evidence": scale_evidence,
                    "enlargement_uniqueness": uniqueness,
                    "enlargement_score": scale_evidence,
                })

        def semantic_metrics(item: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
            box = item.get("prepared_bbox")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                return None
            try:
                left, top, right, bottom = [float(value) for value in box[:4]]
            except (TypeError, ValueError):
                return None
            width = max(1.0, right - left)
            height = max(1.0, bottom - top)
            return width, height, width / height, (left + right) / 2.0, (top + bottom) / 2.0, left

        def sibling_compatible(item: dict[str, Any], members: list[dict[str, Any]]) -> tuple[bool, str]:
            metrics = semantic_metrics(item)
            member_metrics = [semantic_metrics(member) for member in members]
            if metrics is None or any(value is None for value in member_metrics):
                return False, "invalid_geometry"
            widths = [value[0] for value in member_metrics if value is not None]
            heights = [value[1] for value in member_metrics if value is not None]
            aspects = [value[2] for value in member_metrics if value is not None]
            median_width = median(widths)
            median_height = median(heights)
            median_aspect = median(aspects)
            width_ratio = metrics[0] / max(median_width, 1e-6)
            height_ratio = metrics[1] / max(median_height, 1e-6)
            if not cls.ENLARGEMENT_SIBLING_MIN_WIDTH_RATIO <= width_ratio <= cls.ENLARGEMENT_SIBLING_MAX_WIDTH_RATIO:
                return False, "width_ratio_mismatch"
            if not cls.ENLARGEMENT_SIBLING_MIN_HEIGHT_RATIO <= height_ratio <= cls.ENLARGEMENT_SIBLING_MAX_HEIGHT_RATIO:
                return False, "height_ratio_mismatch"
            if abs(math.log(max(metrics[2], 1e-6) / max(median_aspect, 1e-6))) > cls.ENLARGEMENT_SIBLING_MAX_ASPECT_LOG_DELTA:
                return False, "aspect_ratio_mismatch"
            member_box = members[0].get("prepared_bbox")
            if not isinstance(member_box, (list, tuple)) or len(member_box) < 4:
                return False, "invalid_geometry"
            same_row = False
            same_column = False
            item_box = item.get("prepared_bbox")
            for member in members:
                other_box = member.get("prepared_bbox")
                if not isinstance(item_box, (list, tuple)) or not isinstance(other_box, (list, tuple)) or len(item_box) < 4 or len(other_box) < 4:
                    continue
                overlap_y = max(0.0, min(float(item_box[3]), float(other_box[3])) - max(float(item_box[1]), float(other_box[1]))) / max(min(metrics[1], semantic_metrics(member)[1]), 1.0)
                overlap_x = max(0.0, min(float(item_box[2]), float(other_box[2])) - max(float(item_box[0]), float(other_box[0]))) / max(min(metrics[0], semantic_metrics(member)[0]), 1.0)
                same_row = same_row or overlap_y >= cls.ENLARGEMENT_SIBLING_MIN_AXIS_OVERLAP
                same_column = same_column or overlap_x >= cls.ENLARGEMENT_SIBLING_MIN_AXIS_OVERLAP
            return (same_row or same_column), "row_column_mismatch" if not (same_row or same_column) else ""

        sibling_sets: list[list[int]] = []
        sibling_group_by_index: dict[int, int] = {}
        sibling_rejections: dict[int, dict[str, str]] = {}
        for peer_set in peer_sets:
            clusters: list[list[dict[str, Any]]] = []
            for index in sorted(peer_set):
                item = by_index.get(index)
                if item is None:
                    continue
                candidates: list[tuple[float, int, list[dict[str, Any]]]] = []
                for cluster_id, cluster in enumerate(clusters):
                    compatible, reason = sibling_compatible(item, cluster)
                    if compatible:
                        metrics = semantic_metrics(item)
                        medians = [median([semantic_metrics(member)[axis] for member in cluster]) for axis in (0, 1, 2)]
                        distance = sum(abs(math.log(max(metrics[axis], 1e-6) / max(medians[axis], 1e-6))) for axis in range(3)) if metrics else float("inf")
                        candidates.append((distance, cluster_id, cluster))
                    else:
                        sibling_rejections.setdefault(index, {})[str(cluster_id)] = reason
                if candidates:
                    _, _, selected_cluster = min(candidates, key=lambda value: (value[0], value[1]))
                    selected_cluster.append(item)
                else:
                    clusters.append([item])
            for cluster in clusters:
                indices = sorted(int(item["index"]) for item in cluster)
                for item in cluster:
                    index = int(item["index"])
                    item["enlargement_sibling_group_id"] = len(sibling_sets) if len(cluster) >= 2 else -1
                    item["enlargement_sibling_indices"] = indices if len(cluster) >= 2 else []
                    item["enlargement_sibling_count"] = max(0, len(cluster) - 1) if len(cluster) >= 2 else 0
                    item["enlargement_sibling_rejection_reasons"] = sibling_rejections.get(index, {})
                if len(cluster) >= 2:
                    sibling_sets.append(indices)
                    for index in indices:
                        sibling_group_by_index[index] = len(sibling_sets) - 1

        for sibling_set in sibling_sets:
            members = [by_index[index] for index in sibling_set if index in by_index]
            visual_widths = [max(1.0, float(item.get("visual_extent_width", 0.0))) for item in members]
            visual_heights = [max(1.0, float(item.get("visual_extent_height", 0.0))) for item in members]
            visual_areas = [width * height for width, height in zip(visual_widths, visual_heights)]
            base_width, base_height, base_area = median(visual_widths), median(visual_heights), median(visual_areas)
            expansion_widths = []
            expansion_heights = []
            for item in members:
                metrics = semantic_metrics(item)
                expansion_widths.append(float(item.get("visual_extent_width", 0.0)) / max(metrics[0], 1e-6) if metrics else 1.0)
                expansion_heights.append(float(item.get("visual_extent_height", 0.0)) / max(metrics[1], 1e-6) if metrics else 1.0)
            median_expansion_width = median(expansion_widths)
            median_expansion_height = median(expansion_heights)
            for item, width, height, area, expansion_width, expansion_height in zip(members, visual_widths, visual_heights, visual_areas, expansion_widths, expansion_heights):
                relative_visual_width = width / max(base_width, 1e-6)
                relative_visual_height = height / max(base_height, 1e-6)
                relative_visual_area = area / max(base_area, 1e-6)
                width_consistency = math.exp(-abs(math.log(max(expansion_width, 1e-6) / max(median_expansion_width, 1e-6))) / 0.45)
                height_consistency = math.exp(-abs(math.log(max(expansion_height, 1e-6) / max(median_expansion_height, 1e-6))) / 0.45)
                aspect_consistency = math.exp(-abs(math.log(max(relative_visual_width / max(relative_visual_height, 1e-6), 1e-6))) / cls.SCALE_BALANCE_SIGMA)
                footprint_consistency = cls._clamp01((width_consistency * height_consistency * aspect_consistency) ** (1.0 / 3.0))
                footprint_valid = footprint_consistency >= 0.20
                uniform_scale = math.sqrt(max(relative_visual_width * relative_visual_height, 0.0))
                scale_growth = max(0.0, uniform_scale - 1.0)
                width_growth = max(0.0, relative_visual_width - 1.0)
                height_growth = max(0.0, relative_visual_height - 1.0)
                uniform_growth = min(width_growth, height_growth)
                scale_balance = math.exp(-abs(math.log(max(relative_visual_width / max(relative_visual_height, 1e-6), 1e-6))) / cls.SCALE_BALANCE_SIGMA)
                base_score = cls._clamp01(scale_growth / max(cls.ENLARGEMENT_FULL_SCALE_GROWTH, 1e-6))
                balance_gate = cls.ENLARGEMENT_BALANCE_FLOOR + (1.0 - cls.ENLARGEMENT_BALANCE_FLOOR) * scale_balance
                two_axis_support = cls._clamp01(uniform_growth / max(cls.ENLARGEMENT_MIN_MEANINGFUL_GROWTH, 1e-6))
                footprint_gate = 0.5 + 0.5 * footprint_consistency
                extent_symmetry = float(item.get("extent_symmetry", 0.0))
                extent_symmetry_gate = cls.ENLARGEMENT_SYMMETRY_FLOOR + (1.0 - cls.ENLARGEMENT_SYMMETRY_FLOOR) * extent_symmetry
                item.update({
                    "relative_visual_width": relative_visual_width,
                    "relative_visual_height": relative_visual_height,
                    "relative_visual_area": relative_visual_area,
                    "visual_width_growth": width_growth,
                    "visual_height_growth": height_growth,
                    "visual_uniform_growth": uniform_growth,
                    "visual_uniform_scale": uniform_scale,
                    "visual_scale_growth": scale_growth,
                    "visual_scale_balance": scale_balance,
                    "visual_scale_balance_gate": balance_gate,
                    "visual_two_axis_support": two_axis_support,
                    "extent_to_semantic_width_ratio": expansion_width,
                    "extent_to_semantic_height_ratio": expansion_height,
                    "extent_to_semantic_area_ratio": float(item.get("visual_extent_area", 0.0)) / max(metrics[0] * metrics[1], 1e-6) if metrics else 1.0,
                    "relative_extent_expansion_width": relative_visual_width,
                    "relative_extent_expansion_height": relative_visual_height,
                    "extent_uniform_scale": uniform_scale,
                    "extent_scale_growth": scale_growth,
                    "extent_scale_balance": scale_balance,
                    "extent_scale_balance_gate": balance_gate,
                    "extent_two_axis_support": two_axis_support,
                    "extent_symmetry_gate": extent_symmetry_gate,
                    "sibling_median_footprint_width_ratio": median_expansion_width,
                    "sibling_median_footprint_height_ratio": median_expansion_height,
                    "sibling_median_extent_width_ratio": median_expansion_width,
                    "sibling_median_extent_height_ratio": median_expansion_height,
                    "footprint_to_semantic_width_ratio": expansion_width,
                    "footprint_to_semantic_height_ratio": expansion_height,
                    "footprint_consistency_score": footprint_consistency,
                    "footprint_valid": footprint_valid,
                    "enlargement_score": 0.0 if not footprint_valid else cls._clamp01(base_score * balance_gate * two_axis_support * extent_symmetry_gate),
                    "base_enlargement_score": base_score,
                    "scale_evidence": cls._clamp01(base_score * balance_gate * two_axis_support),
                })

        return {"sibling_sets": sibling_sets, "sibling_group_by_index": sibling_group_by_index}

    @classmethod
    def _run_visual_focus_cascade(
        cls,
        evidence: list[dict[str, Any]],
        peer_sets: list[list[int]],
        peer_group_by_index: dict[int, int],
        isolated_indices: list[int],
        enlargement_peer_sets: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        values = {int(item["index"]): item for item in evidence}

        def skipped(stage: str) -> dict[str, Any]:
            return {"stage": stage, "executed": False, "matched": False, "candidate_index": None, "score": None, "runner_up_score": None, "margin": None, "peer_group_id": None, "reason": "skipped_due_to_prior_match"}

        def search(stage: str, field: str, minimum: float, margin_minimum: float, stage_peer_sets: list[list[int]]) -> dict[str, Any]:
            winners = []
            for peer_id, peer_set in enumerate(stage_peer_sets):
                candidates = [values[index] for index in peer_set if index in values]
                if len(candidates) < cls.MIN_COMPARABLE_PEERS + 1:
                    continue
                ranked = sorted(candidates, key=lambda item: (-float(item.get(field, 0.0)), int(item["index"])))
                best = ranked[0]
                runner_score = float(ranked[1].get(field, 0.0)) if len(ranked) > 1 else 0.0
                winners.append((float(best.get(field, 0.0)), float(best.get(field, 0.0)) - runner_score, best, runner_score, peer_id))
            if not winners:
                return {"stage": stage, "executed": True, "matched": False, "candidate_index": None, "score": 0.0, "runner_up_score": 0.0, "margin": 0.0, "peer_group_id": None, "reason": "no_comparable_peer_set"}
            score, margin, best, runner_score, peer_id = max(winners, key=lambda item: (item[0], item[1], -int(item[2]["index"])))
            matched = score >= minimum and margin >= margin_minimum
            confident = sorted(
                [winner for winner in winners if winner[0] >= minimum and winner[1] >= margin_minimum],
                key=lambda item: (-item[0], -item[1], int(item[2]["index"])),
            )
            cross_set_margin = None
            if len(confident) > 1:
                cross_set_margin = confident[0][0] - confident[1][0]
                if cross_set_margin < cls.ENLARGEMENT_V5_MIN_CROSS_SET_MARGIN and stage == "enlargement":
                    matched = False
            return {"stage": stage, "executed": True, "matched": matched, "candidate_index": int(best["index"]), "score": score, "runner_up_score": runner_score, "margin": margin, "cross_set_margin": cross_set_margin, "peer_group_id": peer_id, "reason": "confident_peer_winner" if matched else ("ambiguous_multiple_sibling_sets" if stage == "enlargement" and cross_set_margin is not None else "below_threshold_or_margin")}

        outline = search("outline", "outline_score", cls.OUTLINE_V5_MIN_SCORE, cls.OUTLINE_V5_MIN_MARGIN, peer_sets)
        if outline["matched"]:
            enlargement = skipped("enlargement")
            highlight = skipped("highlight")
            isolated = skipped("isolated_fallback")
            final = outline
        else:
            enlargement = search("enlargement", "enlargement_score", cls.ENLARGEMENT_V5_MIN_SCORE, cls.ENLARGEMENT_V5_MIN_MARGIN, enlargement_peer_sets or [])
            if enlargement["matched"]:
                highlight = skipped("highlight")
                isolated = skipped("isolated_fallback")
                final = enlargement
            else:
                highlight = search("highlight", "highlight_score", cls.HIGHLIGHT_V5_MIN_SCORE, cls.HIGHLIGHT_V5_MIN_MARGIN, peer_sets)
                if highlight["matched"]:
                    isolated = skipped("isolated_fallback")
                    final = highlight
                else:
                    isolated_values = [values[index] for index in isolated_indices if index in values]
                    best_isolated = max(isolated_values, key=lambda item: (float(item.get("direct_visual_confidence", 0.0)), -int(item["index"])), default=None)
                    isolated = {"stage": "isolated_fallback", "executed": True, "matched": False, "candidate_index": int(best_isolated["index"]) if best_isolated else None, "score": float(best_isolated.get("direct_visual_confidence", 0.0)) if best_isolated else 0.0, "runner_up_score": None, "margin": None, "peer_group_id": -1 if best_isolated else None, "reason": "isolated_candidates_observed_no_auto_match" if best_isolated else "no_isolated_candidate"}
                    final = {"stage": "isolated_fallback", "matched": False, "candidate_index": None, "score": isolated["score"], "margin": None, "peer_group_id": None, "reason": "no_peer_match"}
        return {
            "outline_decision": outline,
            "enlargement_decision": enlargement,
            "highlight_decision": highlight,
            "isolated_decision": isolated,
            "focus_visual_v5_stage": final["stage"],
            "focus_visual_v5_matched": bool(final["matched"]),
            "focus_visual_v5_candidate_index": final.get("candidate_index"),
            "focus_visual_v5_score": final.get("score"),
            "focus_visual_v5_margin": final.get("margin"),
            "focus_visual_v5_peer_group_id": final.get("peer_group_id"),
            "focus_visual_v5_reason": final.get("reason"),
        }

    @classmethod
    def _save_peer_debug_image(
        cls,
        image: Image.Image,
        prepared_candidate_bboxes: dict[int, list[int]],
        peer_analysis: dict[str, Any],
        enlargement_sibling_analysis: dict[str, Any] | None = None,
    ) -> None:
        debug_image = image.convert("RGB").copy()
        draw = ImageDraw.Draw(debug_image)
        peer_by_index = peer_analysis["peer_group_by_index"]
        for item in peer_analysis["debug"]:
            index = int(item["index"])
            bbox = prepared_candidate_bboxes.get(index)
            if not bbox or len(bbox) < 4:
                continue
            left, top, right, bottom = [int(round(float(value))) for value in bbox[:4]]
            peer_id = peer_by_index.get(index, -1)
            sibling_id = (enlargement_sibling_analysis or {}).get("sibling_group_by_index", {}).get(index, -1)
            if peer_id >= 0:
                label = f"P{peer_id}/S{sibling_id}:#{index}" if sibling_id >= 0 else f"P{peer_id}/S-:#{index}"
            else:
                label = f"ISO:#{index}"
            color = (30, 150, 255) if peer_id >= 0 else (230, 120, 30)
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            label_top = max(0, top - text_height - 4)
            draw.rectangle((left, label_top, left + text_width + 6, label_top + text_height + 4), fill=(0, 0, 0))
            draw.text((left + 3, label_top + 2), label, fill=color)
        debug_image.save(cls.DEBUG_PEER_IMAGE_PATH, format="JPEG", quality=95)

    @classmethod
    def _visual_focus_evidence(
        cls,
        image: Image.Image,
        candidates: list[dict[str, Any]],
        candidate_groups: list[list[int]],
        prepared_candidate_bboxes: dict[int, list[int]],
        comparison_groups: list[list[int]] | None = None,
        peer_group_by_index: dict[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Compute diagnostic-only focus decoration evidence from the source image."""
        geometries = {
            index: tuple(bbox)
            if isinstance(bbox, list) and len(bbox) == 4
            else None
            for index, bbox in (
                (index, prepared_candidate_bboxes.get(index))
                for index in range(len(candidates))
            )
        }
        group_by_index = {
            index: group_id
            for group_id, group in enumerate(candidate_groups)
            for index in group
        }
        visual_cells = {
            index: cls._peer_visual_cell(index, group, geometries, image)
            for group in candidate_groups
            for index in group
            if index in geometries and geometries[index] is not None
        }
        raw: dict[int, dict[str, float]] = {}
        for index, geometry in geometries.items():
            group_id = group_by_index.get(index, -1)
            if geometry is None:
                raw[index] = {
                    "visual_cell": None,
                    "selected_container": None,
                    "expansion_ratio": 0.0,
                    "proposals": [],
                    "raw_score": 0.0,
                    "continuity": 0.0,
                    "outer_ring": 0.0,
                    "background": 0.0,
                    "edge": 0.0,
                    "area": 1.0,
                    "width": 1.0,
                    "height": 1.0,
                }
                continue
            left, top, right, bottom = geometry
            width = max(1, right - left)
            height = max(1, bottom - top)
            cell = visual_cells.get(index)
            proposals = cls._container_proposals(
                image, index, geometry, cell, candidate_groups[group_id]
                if group_id >= 0 else [index], geometries
            )
            scored_proposals: list[dict[str, Any]] = []
            for proposal in proposals:
                ring_continuity, ring_contrast, background_delta, edge_strength, side_support = (
                    cls._decoration_features(image, proposal["bbox"])
                )
                decoration_score = (
                    cls.VISUAL_RING_CONTINUITY_WEIGHT * ring_continuity
                    + cls.VISUAL_RING_CONTRAST_WEIGHT * ring_contrast
                    + cls.VISUAL_BACKGROUND_WEIGHT * background_delta
                )
                expansion = proposal["expansion_ratio"]
                scale_penalty = max(0.75, 1.0 - 0.5 * expansion)
                scored_proposals.append({
                    **proposal,
                    "ring_continuity": ring_continuity,
                    "outer_ring_contrast": ring_contrast,
                    "background_highlight_evidence": background_delta,
                    "highlight_side_support": side_support,
                    "score": decoration_score * scale_penalty,
                    "raw_score": decoration_score,
                    "edge": edge_strength,
                })
            best = max(
                scored_proposals,
                key=lambda item: (item["score"], -item["expansion_ratio"]),
            ) if scored_proposals else {
                "bbox": geometry,
                "expansion_ratio": 0.0,
                "ring_continuity": 0.0,
                "outer_ring_contrast": 0.0,
                "background_highlight_evidence": 0.0,
                "score": 0.0,
                "raw_score": 0.0,
                "edge": 0.0,
            }
            raw[index] = {
                "visual_cell": cell,
                "selected_container": best["bbox"],
                "expansion_ratio": best["expansion_ratio"],
                "proposals": scored_proposals,
                "raw_score": best["raw_score"],
                "continuity": best["ring_continuity"],
                "outer_ring": best["outer_ring_contrast"],
                "background": best["background_highlight_evidence"],
                "side_support": best.get("highlight_side_support", {}),
                "edge": best["edge"],
                "area": float(width * height),
                "width": float(width),
                "height": float(height),
                "group_id": float(group_id),
            }

        absolute_outline_scores = {
            index: cls._clamp01(
                raw[index]["continuity"] * math.sqrt(raw[index]["outer_ring"])
            )
            for index in raw
        }
        evidence: list[dict[str, Any]] = []
        for index in range(len(candidates)):
            values = raw[index]
            peer_indices = [
                peer for peer in candidate_groups[group_by_index[index]]
                if peer in raw and geometries.get(peer) is not None
            ] if index in group_by_index else [index]
            if not peer_indices:
                peer_indices = [index]
            continuity = cls._peer_relative(
                values["continuity"],
                [raw[peer]["continuity"] for peer in peer_indices],
            )
            ring = cls._peer_relative(values["outer_ring"], [raw[peer]["outer_ring"] for peer in peer_indices])
            background = cls._peer_relative(values["background"], [raw[peer]["background"] for peer in peer_indices])
            edge = values["edge"]
            peer_areas = [raw[peer]["area"] for peer in peer_indices]
            peer_widths = [raw[peer]["width"] for peer in peer_indices]
            peer_heights = [raw[peer]["height"] for peer in peer_indices]
            size_ratio = values["area"] / max(cls._median(peer_areas), 1.0)
            width_ratio = values["width"] / max(cls._median(peer_widths), 1.0)
            height_ratio = values["height"] / max(cls._median(peer_heights), 1.0)
            absolute_outline_score = absolute_outline_scores[index]
            other_outline_scores = [
                absolute_outline_scores[peer]
                for peer in peer_indices
                if peer != index and peer in absolute_outline_scores
            ]
            outline_exclusivity = cls._outline_exclusivity(
                absolute_outline_score, other_outline_scores
            )
            outline_exclusivity_gate = (
                cls.OUTLINE_EXCLUSIVITY_FLOOR
                + (1.0 - cls.OUTLINE_EXCLUSIVITY_FLOOR) * outline_exclusivity
            )
            outline_score = cls._clamp01(
                absolute_outline_score * outline_exclusivity_gate
            )
            highlight_side_support = values.get("side_support", {})
            highlight_consistency = cls._clamp01(
                sum(highlight_side_support.values()) / 4.0
                if highlight_side_support else 0.0
            )
            highlight_score = cls._clamp01(background * highlight_consistency)
            enlargement_score, protrusion, size_consistency, scale_details = (
                cls._enlargement_evidence(index, peer_indices, raw, geometries)
            )
            direct_visual_confidence = max(
                outline_score, highlight_score, enlargement_score
            )
            evidence.append({
                "index": index,
                "group_id": group_by_index.get(index, -1),
                "visual_cell_bbox": values["visual_cell"],
                "selected_container_bbox": values["selected_container"],
                "container_expansion_ratio": round(values["expansion_ratio"], 4),
                "raw_decoration_score": round(values["raw_score"], 4),
                "prepared_bbox": (
                    list(geometries[index]) if geometries[index] is not None else None
                ),
                "prepared_candidate_width": (
                    geometries[index][2] - geometries[index][0]
                    if geometries[index] is not None else 0
                ),
                "prepared_candidate_height": (
                    geometries[index][3] - geometries[index][1]
                    if geometries[index] is not None else 0
                ),
                "ring_continuity": round(continuity, 4),
                "outer_ring_contrast": round(ring, 4),
                "background_highlight_evidence": round(background, 4),
                "absolute_outline_score": round(absolute_outline_score, 4),
                "raw_outline_score": round(absolute_outline_score, 4),
                "outline_exclusivity": round(outline_exclusivity, 4),
                "outline_exclusivity_gate": round(outline_exclusivity_gate, 4),
                "outline_score": round(outline_score, 4),
                "highlight_score": round(highlight_score, 4),
                "enlargement_score": round(enlargement_score, 4),
                "size_ratio": round(size_ratio, 4),
                "width_ratio": round(width_ratio, 4),
                "height_ratio": round(height_ratio, 4),
                "local_edge_strength": round(edge, 4),
                "relative_width": round(scale_details["relative_width"], 4),
                "relative_height": round(scale_details["relative_height"], 4),
                "relative_area": round(scale_details["relative_area"], 4),
                "peer_size_consistency": round(size_consistency, 4),
                "peer_protrusion_score": round(protrusion, 4),
                "width_growth": round(scale_details["width_growth"], 4),
                "height_growth": round(scale_details["height_growth"], 4),
                "uniform_growth": round(scale_details["uniform_growth"], 4),
                "scale_balance": round(scale_details["scale_balance"], 4),
                "scale_evidence": round(scale_details["scale_evidence"], 4),
                "enlargement_uniqueness": round(
                    scale_details["enlargement_uniqueness"], 4
                ),
                "outline_side_support": scale_details["outline_side_support"],
                "highlight_side_support": {
                    side: round(value, 4)
                    for side, value in highlight_side_support.items()
                },
                "direct_visual_confidence": round(direct_visual_confidence, 4),
                "visual_focus_score": round(direct_visual_confidence, 4),
                "container_proposals": [
                    {
                        "bbox": list(item["bbox"]),
                        "expansion_ratio": round(item["expansion_ratio"], 4),
                        "ring_continuity": round(item["ring_continuity"], 4),
                        "outer_ring_contrast": round(item["outer_ring_contrast"], 4),
                        "background_highlight_evidence": round(
                            item["background_highlight_evidence"], 4
                        ),
                        "raw_score": round(item["raw_score"], 4),
                    }
                    for item in values["proposals"]
                ],
            })
        return evidence

    @staticmethod
    def _candidate_pixel_geometry(image: Image.Image, candidate: Any) -> tuple[int, int, int, int] | None:
        geometry = FocusResolver._candidate_geometry(candidate)
        if geometry is None:
            return None
        _, _, _, _, left, top, right, bottom = geometry
        return (
            max(0, min(image.width - 1, round(left * image.width / 1000))),
            max(0, min(image.height - 1, round(top * image.height / 1000))),
            max(1, min(image.width, round(right * image.width / 1000))),
            max(1, min(image.height, round(bottom * image.height / 1000))),
        )

    @classmethod
    def _peer_visual_cell(
        cls,
        index: int,
        group: list[int],
        geometries: dict[int, tuple[int, int, int, int] | None],
        image: Image.Image,
    ) -> list[int] | None:
        geometry = geometries.get(index)
        if geometry is None:
            return None
        left, top, right, bottom = geometry
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        width = max(1, right - left)
        height = max(1, bottom - top)
        peers = [
            geometries[peer]
            for peer in group
            if peer != index and geometries.get(peer) is not None
        ]
        row_peers = [
            peer for peer in peers
            if max(0, min(bottom, peer[3]) - max(top, peer[1]))
            / max(1, min(height, peer[3] - peer[1])) >= 0.35
            or abs(center_y - (peer[1] + peer[3]) / 2.0)
            <= 0.65 * max(height, peer[3] - peer[1])
        ]
        column_peers = [
            peer for peer in peers
            if max(0, min(right, peer[2]) - max(left, peer[0]))
            / max(1, min(width, peer[2] - peer[0])) >= 0.35
            or abs(center_x - (peer[0] + peer[2]) / 2.0)
            <= 0.65 * max(width, peer[2] - peer[0])
        ]
        row_peers.sort(key=lambda peer: (peer[0] + peer[2]) / 2.0)
        column_peers.sort(key=lambda peer: (peer[1] + peer[3]) / 2.0)
        previous_row = [peer for peer in row_peers if (peer[0] + peer[2]) / 2.0 < center_x]
        next_row = [peer for peer in row_peers if (peer[0] + peer[2]) / 2.0 > center_x]
        previous_column = [peer for peer in column_peers if (peer[1] + peer[3]) / 2.0 < center_y]
        next_column = [peer for peer in column_peers if (peer[1] + peer[3]) / 2.0 > center_y]
        cell_left = (
            round((previous_row[-1][0] + previous_row[-1][2]) / 2.0)
            if previous_row else left - max(1, round(width * 0.5))
        )
        cell_right = (
            round((next_row[0][0] + next_row[0][2]) / 2.0)
            if next_row else right + max(1, round(width * 0.5))
        )
        cell_top = (
            round((previous_column[-1][1] + previous_column[-1][3]) / 2.0)
            if previous_column else top - max(1, round(height * 0.5))
        )
        cell_bottom = (
            round((next_column[0][1] + next_column[0][3]) / 2.0)
            if next_column else bottom + max(1, round(height * 0.5))
        )
        return [
            max(0, min(cell_left, left)),
            max(0, min(cell_top, top)),
            min(image.width, max(cell_right, right)),
            min(image.height, max(cell_bottom, bottom)),
        ]

    @classmethod
    def _container_proposals(
        cls,
        image: Image.Image,
        candidate_index: int,
        geometry: tuple[int, int, int, int],
        cell: list[int] | None,
        group: list[int],
        geometries: dict[int, tuple[int, int, int, int] | None],
    ) -> list[dict[str, Any]]:
        if cell is None:
            return []
        left, top, right, bottom = geometry
        width = max(1, right - left)
        height = max(1, bottom - top)
        peer_centers = [
            ((peer[0] + peer[2]) / 2.0, (peer[1] + peer[3]) / 2.0)
            for index in group
            if index in geometries and geometries[index] is not None
            and index != candidate_index
            for peer in [geometries[index]]
        ]
        proposals: list[dict[str, Any]] = []
        for expansion in cls.CONTAINER_PROPOSAL_EXPANSIONS:
            horizontal = max(1 if expansion > 0 else 0, round(width * expansion))
            vertical = max(1 if expansion > 0 else 0, round(height * expansion))
            proposal = (
                max(cell[0], left - horizontal),
                max(cell[1], top - vertical),
                min(cell[2], right + horizontal),
                min(cell[3], bottom + vertical),
            )
            if proposal[0] > left or proposal[1] > top or proposal[2] < right or proposal[3] < bottom:
                continue
            proposal_center = ((proposal[0] + proposal[2]) / 2.0, (proposal[1] + proposal[3]) / 2.0)
            original_center = ((left + right) / 2.0, (top + bottom) / 2.0)
            center_distance = math.hypot(
                proposal_center[0] - original_center[0],
                proposal_center[1] - original_center[1],
            )
            if center_distance > 0.25 * max(width, height):
                continue
            if any(
                proposal[0] <= center_x < proposal[2]
                and proposal[1] <= center_y < proposal[3]
                for center_x, center_y in peer_centers
            ):
                continue
            actual_expansion = max(
                (proposal[2] - proposal[0]) / width - 1.0,
                (proposal[3] - proposal[1]) / height - 1.0,
                0.0,
            )
            proposals.append({
                "bbox": proposal,
                "expansion_ratio": actual_expansion,
            })
        return proposals

    @classmethod
    def _decoration_features(
        cls,
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float, dict[str, float]]:
        ring_continuity, ring_contrast = cls._perimeter_ring_evidence(image, box)
        background_mean = cls._mean_luma_band(image, box, 14, 5)
        perimeter_mean = cls._perimeter_luma(image, box)
        background_delta = cls._clamp01(
            abs(perimeter_mean - background_mean) / 255.0
        )
        edge_strength = cls._perimeter_edge_strength(image, box)
        return (
            ring_continuity,
            ring_contrast,
            background_delta,
            edge_strength,
            cls._highlight_side_support(image, box),
        )

    @classmethod
    def _highlight_side_support(
        cls,
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> dict[str, float]:
        left, top, right, bottom = box
        near = max(1, min(4, round(min(right - left, bottom - top) * 0.02)))
        far = max(near + 2, min(24, round(min(right - left, bottom - top) * 0.08)))
        side_values: dict[str, list[float]] = {
            "top": [], "bottom": [], "left": [], "right": []
        }
        count = 24
        for position in range(count):
            fraction = (position + 0.5) / count
            x = round(left + fraction * max(1, right - left - 1))
            y = round(top + fraction * max(1, bottom - top - 1))
            if top >= far:
                side_values["top"].append(cls._clamp01(
                    abs(
                        cls._sample_luma(image, (x, top + near), True)
                        - cls._sample_luma(image, (x, top - far), True)
                    ) / 255.0
                ))
            if bottom + far < image.height:
                side_values["bottom"].append(cls._clamp01(
                    abs(
                        cls._sample_luma(image, (x, bottom - near - 1), True)
                        - cls._sample_luma(image, (x, bottom + far), True)
                    ) / 255.0
                ))
            if left >= far:
                side_values["left"].append(cls._clamp01(
                    abs(
                        cls._sample_luma(image, (left + near, y), False)
                        - cls._sample_luma(image, (left - far, y), False)
                    ) / 255.0
                ))
            if right + far < image.width:
                side_values["right"].append(cls._clamp01(
                    abs(
                        cls._sample_luma(image, (right - near - 1, y), False)
                        - cls._sample_luma(image, (right + far, y), False)
                    ) / 255.0
                ))
        return {
            side: cls._clamp01(sum(values) / len(values)) if values else 0.0
            for side, values in side_values.items()
        }

    @staticmethod
    def _pixel_luma(pixel: Any) -> float:
        red, green, blue = pixel[:3]
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    @classmethod
    def _mean_luma_band(
        cls,
        image: Image.Image,
        box: tuple[int, int, int, int],
        outer_margin: int,
        inner_margin: int,
    ) -> float:
        left, top, right, bottom = box
        outer = (
            max(0, left - outer_margin), max(0, top - outer_margin),
            min(image.width, right + outer_margin), min(image.height, bottom + outer_margin),
        )
        inner = (
            max(0, left - inner_margin), max(0, top - inner_margin),
            min(image.width, right + inner_margin), min(image.height, bottom + inner_margin),
        )
        return cls._mean_luma_region(image, outer, inner)

    @classmethod
    def _mean_luma_region(
        cls,
        image: Image.Image,
        box: tuple[int, int, int, int],
        excluded: tuple[int, int, int, int] | None = None,
    ) -> float:
        left, top, right, bottom = box
        area = max(1, (right - left) * (bottom - top))
        stride = max(1, round(math.sqrt(area / 2500)))
        total = 0.0
        count = 0
        for y in range(top, bottom, stride):
            for x in range(left, right, stride):
                if excluded is not None and excluded[0] <= x < excluded[2] and excluded[1] <= y < excluded[3]:
                    continue
                total += cls._pixel_luma(image.getpixel((x, y)))
                count += 1
        return total / count if count else 0.0

    @classmethod
    def _perimeter_ring_evidence(
        cls,
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        """Measure sustained intermediate decoration around a candidate."""
        left, top, right, bottom = box
        samples: list[tuple[int, bool, float]] = []
        sample_count = 32
        threshold = 0.10
        candidate_width = max(1, right - left)
        candidate_height = max(1, bottom - top)
        near_distance = max(
            1, min(4, round(min(candidate_width, candidate_height) * 0.02))
        )
        far_distance = max(
            near_distance + 2,
            min(24, round(min(candidate_width, candidate_height) * 0.08)),
        )

        def add_sample(
            side: int,
            position: int,
            inner: tuple[int, int],
            immediate: tuple[int, int],
            farther: tuple[int, int],
            horizontal: bool,
        ) -> None:
            if not all(
                0 <= point[0] < image.width and 0 <= point[1] < image.height
                for point in (inner, immediate, farther)
            ):
                return
            inner_luma = cls._sample_luma(image, inner, horizontal)
            immediate_luma = cls._sample_luma(image, immediate, horizontal)
            farther_luma = cls._sample_luma(image, farther, horizontal)
            inner_delta = abs(immediate_luma - inner_luma) / 255.0
            background_delta = abs(immediate_luma - farther_luma) / 255.0
            contrast = min(inner_delta, background_delta)
            samples.append((side, contrast >= threshold, contrast))

        for position in range(sample_count):
            fraction = (position + 0.5) / sample_count
            x = round(left + fraction * max(1, right - left - 1))
            y = round(top + fraction * max(1, bottom - top - 1))
            if top >= far_distance:
                add_sample(
                    0, position,
                    (x, top + near_distance),
                    (x, top - near_distance),
                    (x, top - far_distance),
                    True,
                )
            if bottom + far_distance < image.height:
                add_sample(
                    1, position,
                    (x, bottom - near_distance - 1),
                    (x, bottom + near_distance),
                    (x, bottom + far_distance),
                    True,
                )
            if left >= far_distance:
                add_sample(
                    2, position,
                    (left + near_distance, y),
                    (left - near_distance, y),
                    (left - far_distance, y),
                    False,
                )
            if right + far_distance < image.width:
                add_sample(
                    3, position,
                    (right - near_distance - 1, y),
                    (right + near_distance, y),
                    (right + far_distance, y),
                    False,
                )

        if not samples:
            return 0.0, 0.0
        positive_fraction = sum(positive for _, positive, _ in samples) / len(samples)
        side_runs: list[float] = []
        for side in range(4):
            side_samples = [positive for sample_side, positive, _ in samples if sample_side == side]
            if not side_samples:
                continue
            longest = current = 0
            for positive in side_samples:
                current = current + 1 if positive else 0
                longest = max(longest, current)
            side_runs.append(longest / len(side_samples))
        sustained_factor = 0.5 + 0.5 * min(1.0, 2.0 * max(side_runs, default=0.0))
        continuity = cls._clamp01(positive_fraction * sustained_factor)
        contrast = sum(value for _, _, value in samples) / len(samples)
        return continuity, cls._clamp01(contrast)

    @classmethod
    def _sample_luma(
        cls,
        image: Image.Image,
        point: tuple[int, int],
        horizontal: bool,
    ) -> float:
        x, y = point
        tangent_points = (
            ((x - 1, y), (x, y), (x + 1, y))
            if horizontal
            else ((x, y - 1), (x, y), (x, y + 1))
        )
        valid = [
            cls._pixel_luma(image.getpixel(sample))
            for sample in tangent_points
            if 0 <= sample[0] < image.width and 0 <= sample[1] < image.height
        ]
        return sum(valid) / len(valid) if valid else 0.0

    @classmethod
    def _perimeter_luma(cls, image: Image.Image, box: tuple[int, int, int, int]) -> float:
        left, top, right, bottom = box
        inset_x = max(1, (right - left) // 5)
        inset_y = max(1, (bottom - top) // 5)
        return cls._mean_luma_region(
            image,
            box,
            (left + inset_x, top + inset_y, max(left + inset_x, right - inset_x), max(top + inset_y, bottom - inset_y)),
        )

    @classmethod
    def _perimeter_edge_strength(cls, image: Image.Image, box: tuple[int, int, int, int]) -> float:
        left, top, right, bottom = box
        samples: list[float] = []
        for x in range(left, right, max(1, (right - left) // 80)):
            if top > 0 and top < image.height:
                samples.append(abs(cls._pixel_luma(image.getpixel((x, top))) - cls._pixel_luma(image.getpixel((x, top - 1)))) / 255.0)
            if bottom > 0 and bottom < image.height:
                samples.append(abs(cls._pixel_luma(image.getpixel((x, bottom - 1))) - cls._pixel_luma(image.getpixel((x, bottom)))) / 255.0)
        for y in range(top, bottom, max(1, (bottom - top) // 80)):
            if left > 0 and left < image.width:
                samples.append(abs(cls._pixel_luma(image.getpixel((left, y))) - cls._pixel_luma(image.getpixel((left - 1, y)))) / 255.0)
            if right > 0 and right < image.width:
                samples.append(abs(cls._pixel_luma(image.getpixel((right - 1, y))) - cls._pixel_luma(image.getpixel((right, y)))) / 255.0)
        return sum(samples) / len(samples) if samples else 0.0

    @classmethod
    def _peer_relative(cls, value: float, peers: list[float]) -> float:
        if len(peers) <= 1:
            return cls._clamp01(value)
        median = cls._median(peers)
        return cls._clamp01((value - median) / max(median, 0.05))

    @classmethod
    def _enlargement_evidence(
        cls,
        index: int,
        peer_indices: list[int],
        raw: dict[int, dict[str, Any]],
        geometries: dict[int, tuple[int, int, int, int] | None],
    ) -> tuple[float, float, float, dict[str, Any]]:
        details = {
            "relative_width": 1.0,
            "relative_height": 1.0,
            "relative_area": 1.0,
            "width_growth": 0.0,
            "height_growth": 0.0,
            "uniform_growth": 0.0,
            "scale_balance": 0.0,
            "scale_evidence": 0.0,
            "enlargement_uniqueness": 0.0,
            "outline_side_support": {},
        }
        others = [peer for peer in peer_indices if peer != index and peer in raw]
        if not others:
            return 0.0, 0.0, 0.0, details
        peer_widths = [raw[peer]["width"] for peer in peer_indices]
        peer_heights = [raw[peer]["height"] for peer in peer_indices]
        peer_areas = [raw[peer]["area"] for peer in peer_indices]
        relative_width = raw[index]["width"] / max(cls._median(peer_widths), 1.0)
        relative_height = raw[index]["height"] / max(cls._median(peer_heights), 1.0)
        relative_area = raw[index]["area"] / max(cls._median(peer_areas), 1.0)
        width_growth = max(0.0, relative_width - 1.0)
        height_growth = max(0.0, relative_height - 1.0)
        uniform_growth = min(width_growth, height_growth)
        safe_width = max(relative_width, 1e-6)
        safe_height = max(relative_height, 1e-6)
        scale_balance = math.exp(
            -abs(math.log(safe_width / safe_height)) / cls.SCALE_BALANCE_SIGMA
        )
        scale_balance = cls._clamp01(
            scale_balance if math.isfinite(scale_balance) else 0.0
        )
        details.update({
            "relative_width": relative_width,
            "relative_height": relative_height,
            "relative_area": relative_area,
        })
        log_widths = [math.log(max(raw[peer]["width"], 1.0)) for peer in peer_indices]
        log_heights = [math.log(max(raw[peer]["height"], 1.0)) for peer in peer_indices]
        log_areas = [math.log(max(raw[peer]["area"], 1.0)) for peer in peer_indices]
        consistency_deviation = cls._median([
            abs(log_widths[position] - cls._median(log_widths))
            + abs(log_heights[position] - cls._median(log_heights))
            + abs(log_areas[position] - cls._median(log_areas))
            for position in range(len(peer_indices))
        ]) / 3.0
        consistency = cls._clamp01(1.0 - consistency_deviation / 0.35)

        area_uniqueness = cls._positive_uniqueness(
            raw[index]["area"], [raw[peer]["area"] for peer in others]
        )
        width_uniqueness = cls._positive_uniqueness(
            raw[index]["width"], [raw[peer]["width"] for peer in others]
        )
        height_uniqueness = cls._positive_uniqueness(
            raw[index]["height"], [raw[peer]["height"] for peer in others]
        )
        uniqueness = cls._clamp01(
            (width_uniqueness + height_uniqueness + area_uniqueness) / 3.0
        )
        uniform_growth_support = cls._clamp01(1.0 - math.exp(-3.0 * uniform_growth))
        area_support = cls._clamp01(1.0 - 1.0 / max(relative_area, 1.0))
        scale_evidence = cls._clamp01(
            (0.75 * uniform_growth_support + 0.25 * area_support)
            * scale_balance
        )
        protrusion = cls._peer_protrusion_score(index, peer_indices, geometries)
        structural_support = 0.5 + 0.5 * protrusion
        enlargement = cls._clamp01(
            scale_evidence * uniqueness * structural_support * consistency
        )
        details.update({
            "width_growth": width_growth,
            "height_growth": height_growth,
            "uniform_growth": uniform_growth,
            "scale_balance": scale_balance,
            "scale_evidence": scale_evidence,
            "enlargement_uniqueness": uniqueness,
        })
        return enlargement, protrusion, consistency, details

    @classmethod
    def _outline_exclusivity(
        cls,
        raw_outline_score: float,
        other_peer_scores: list[float],
    ) -> float:
        if not other_peer_scores:
            # A singleton has no comparative evidence; retain a conservative
            # amount of direct evidence without making it highly exclusive.
            return 0.35
        peer_median = cls._median(other_peer_scores)
        peer_mad = cls._median([
            abs(score - peer_median) for score in other_peer_scores
        ])
        return cls._clamp01(
            max(0.0, raw_outline_score - peer_median)
            / max(0.05, 3.0 * peer_mad + 0.02)
        )

    @classmethod
    def _positive_uniqueness(
        cls,
        value: float,
        peer_values: list[float],
    ) -> float:
        if not peer_values:
            return 0.0
        logs = [math.log(max(peer, 1.0)) for peer in peer_values]
        median = cls._median(logs)
        mad = cls._median([abs(item - median) for item in logs])
        positive_difference = max(0.0, math.log(max(value, 1.0)) - median)
        return cls._clamp01(positive_difference / max(0.15, 3.0 * mad + 0.05))

    @classmethod
    def _peer_protrusion_score(
        cls,
        index: int,
        peer_indices: list[int],
        geometries: dict[int, tuple[int, int, int, int] | None],
    ) -> float:
        geometry = geometries.get(index)
        peers = [
            geometries[peer]
            for peer in peer_indices
            if peer != index and geometries.get(peer) is not None
        ]
        if geometry is None or not peers:
            return 0.0
        left, top, right, bottom = geometry
        width = max(1, right - left)
        height = max(1, bottom - top)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        row_peers = [
            peer for peer in peers
            if max(0, min(bottom, peer[3]) - max(top, peer[1]))
            / max(1, min(height, peer[3] - peer[1])) >= 0.35
            or abs(center_y - (peer[1] + peer[3]) / 2.0)
            <= 0.65 * max(height, peer[3] - peer[1])
        ]
        column_peers = [
            peer for peer in peers
            if max(0, min(right, peer[2]) - max(left, peer[0]))
            / max(1, min(width, peer[2] - peer[0])) >= 0.35
            or abs(center_x - (peer[0] + peer[2]) / 2.0)
            <= 0.65 * max(width, peer[2] - peer[0])
        ]
        protrusions: list[float] = []
        if row_peers:
            median_top = cls._median([peer[1] for peer in row_peers])
            median_bottom = cls._median([peer[3] for peer in row_peers])
            protrusions.extend([
                max(0.0, (median_top - top) / height),
                max(0.0, (bottom - median_bottom) / height),
            ])
        if column_peers:
            median_left = cls._median([peer[0] for peer in column_peers])
            median_right = cls._median([peer[2] for peer in column_peers])
            protrusions.extend([
                max(0.0, (median_left - left) / width),
                max(0.0, (right - median_right) / width),
            ])
        return cls._clamp01(2.0 * sum(protrusions) / len(protrusions)) if protrusions else 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _focus_candidate_indices(cls, candidates: list[dict[str, Any]]) -> list[int]:
        """Return every candidate; geometry no longer performs hard filtering."""
        return list(range(len(candidates)))

    @classmethod
    def _focus_candidate_groups(cls, candidates: list[dict[str, Any]]) -> list[list[int]]:
        """Build strict horizontal rows, then compatible local grid groups."""
        geometries: dict[int, tuple[float, float, float, float, float, float, float, float]] = {}
        for index, candidate in enumerate(candidates):
            geometry = cls._candidate_geometry(candidate)
            if geometry is not None:
                geometries[index] = geometry

        rows: list[dict[str, Any]] = []
        for index in sorted(geometries, key=lambda item: (geometries[item][3], item)):
            geometry = geometries[index]
            matching_rows = [row for row in rows if cls._row_accepts(row, geometry)]
            if matching_rows:
                row = min(matching_rows, key=lambda item: abs(item["center_y"] - geometry[3]))
                row["indices"].append(index)
                cls._update_row_statistics(row, geometries)
            else:
                row = {"indices": [index], "center_y": geometry[3]}
                cls._update_row_statistics(row, geometries)
                rows.append(row)

        # Rows are merged only when their aggregate geometry describes stable
        # columns and local vertical spacing. This deliberately is not a graph.
        grid_groups: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: item["center_y"]):
            matching = [
                group for group in grid_groups
                if cls._grid_group_accepts(group, row, geometries)
            ]
            if matching:
                group = min(matching, key=lambda item: abs(item["center_y"] - row["center_y"]))
                group["rows"].append(row)
                group["indices"].extend(row["indices"])
                cls._update_grid_statistics(group, geometries)
            else:
                group = {"rows": [row], "indices": list(row["indices"])}
                cls._update_grid_statistics(group, geometries)
                grid_groups.append(group)

        result = [
            group["indices"]
            for group in sorted(grid_groups, key=lambda item: item["center_y"])
        ]
        result.extend([[index] for index in range(len(candidates)) if index not in geometries])
        return result

    @classmethod
    def _focus_candidate_group_types(
        cls, candidates: list[dict[str, Any]], groups: list[list[int]]
    ) -> list[str]:
        types: list[str] = []
        for group in groups:
            if len(group) == 1:
                types.append("singleton")
                continue
            geometries = [cls._candidate_geometry(candidates[index]) for index in group]
            valid = [geometry for geometry in geometries if geometry is not None]
            center_ys = sorted(geometry[3] for geometry in valid)
            has_multiple_rows = any(
                abs(center_ys[position] - center_ys[position - 1])
                > cls.PEER_MAX_CENTER_Y_DISTANCE_RATIO * max(
                    valid[position][1], valid[position - 1][1]
                )
                for position in range(1, len(center_ys))
            )
            types.append("grid" if has_multiple_rows else "horizontal_row")
        return types

    @classmethod
    def _update_row_statistics(
        cls, row: dict[str, Any], geometries: dict[int, tuple[float, ...]]
    ) -> None:
        values = [geometries[index] for index in row["indices"]]
        row["center_y"] = cls._median([value[3] for value in values])
        row["height"] = cls._median([value[1] for value in values])
        row["width"] = cls._median([value[0] for value in values])
        row["left"] = min(value[4] for value in values)
        row["right"] = max(value[6] for value in values)

    @classmethod
    def _row_accepts(cls, row: dict[str, Any], geometry: tuple[float, ...]) -> bool:
        width, height, _, center_y, left, top, right, bottom = geometry
        height_ratio = min(height, row["height"]) / max(height, row["height"])
        row_top = row["center_y"] - row["height"] / 2
        row_bottom = row["center_y"] + row["height"] / 2
        vertical_overlap = max(0.0, min(bottom, row_bottom) - max(top, row_top))
        same_y = vertical_overlap / min(height, row["height"]) >= cls.PEER_MIN_VERTICAL_OVERLAP_RATIO
        same_y = same_y or abs(center_y - row["center_y"]) <= (
            cls.PEER_MAX_CENTER_Y_DISTANCE_RATIO * max(height, row["height"])
        )
        gap = max(row["left"], left) - min(row["right"], right)
        return (
            same_y
            and height_ratio >= cls.PEER_MIN_HEIGHT_RATIO
            and gap <= cls.PEER_MAX_HORIZONTAL_GAP_RATIO * max(width, row["width"])
        )

    @classmethod
    def _grid_group_accepts(
        cls,
        group: dict[str, Any],
        row: dict[str, Any],
        geometries: dict[int, tuple[float, ...]],
    ) -> bool:
        row_bounds = cls._bounds_for_indices(row.get("indices", []), geometries)
        group_indices = group.get("indices")
        if not isinstance(group_indices, list):
            group_indices = [
                index
                for member_row in group.get("rows", [])
                if isinstance(member_row, dict)
                for index in member_row.get("indices", [])
            ]
        group_bounds = cls._bounds_for_indices(group_indices, geometries)
        if row_bounds is None or group_bounds is None:
            return False
        height_ratio = min(row_bounds["height"], group_bounds["height"]) / max(
            row_bounds["height"], group_bounds["height"]
        )
        if height_ratio < cls.PEER_MIN_HEIGHT_RATIO:
            return False
        group_center_ys = [
            geometries[index][3]
            for index in group_indices
            if isinstance(index, int) and index in geometries
        ]
        if not group_center_ys:
            return False
        group_bottom_center_y = max(group_center_ys)
        row_gap = row_bounds["center_y"] - group_bottom_center_y
        if row_gap < 0 or row_gap > cls.PEER_MAX_VERTICAL_GAP_RATIO * max(
            row_bounds["height"], group_bounds["height"]
        ):
            return False
        proposed_top = min(group_bounds["top"], row_bounds["top"])
        proposed_bottom = max(group_bounds["bottom"], row_bounds["bottom"])
        proposed_left = min(group_bounds["left"], row_bounds["left"])
        proposed_right = max(group_bounds["right"], row_bounds["right"])
        if proposed_bottom - proposed_top > cls.GRID_MAX_VERTICAL_SPAN_HEIGHTS * max(
            row_bounds["height"], group_bounds["height"]
        ):
            return False
        if proposed_right - proposed_left > cls.GRID_MAX_HORIZONTAL_SPAN_WIDTHS * max(
            row_bounds["width"], group_bounds["width"]
        ):
            return False
        if group.get("row_gaps"):
            median_gap = cls._median(group["row_gaps"])
            if abs(row_gap - median_gap) > cls.PEER_MAX_VERTICAL_GAP_RATIO * max(
                row_bounds["height"], group_bounds["height"]
            ):
                return False
        # Every new row must agree with every existing row. Matching only one
        # distant member would recreate transitive bridge chaining.
        for prior_row in group["rows"]:
            for index in row["indices"]:
                if index not in geometries:
                    continue
                geometry = geometries[index]
                if not any(
                    prior in geometries
                    and abs(geometry[2] - geometries[prior][2])
                    <= 0.75 * max(geometry[0], geometries[prior][0])
                    for prior in prior_row["indices"]
                ):
                    return False
        return True

    @staticmethod
    def _bounds_for_indices(
        indices: Any,
        geometries: dict[int, tuple[float, ...]],
    ) -> dict[str, float] | None:
        if not isinstance(indices, list):
            return None
        values = [
            geometries[index]
            for index in indices
            if isinstance(index, int) and index in geometries
        ]
        if not values:
            return None
        left = min(value[4] for value in values)
        top = min(value[5] for value in values)
        right = max(value[6] for value in values)
        bottom = max(value[7] for value in values)
        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
            "center_x": (left + right) / 2.0,
            "center_y": (top + bottom) / 2.0,
        }

    @classmethod
    def _update_grid_statistics(
        cls, group: dict[str, Any], geometries: dict[int, tuple[float, ...]]
    ) -> None:
        values = [geometries[index] for index in group["indices"]]
        group["center_y"] = cls._median([value[3] for value in values])
        group["height"] = cls._median([value[1] for value in values])
        group["bottom_center_y"] = max(value[3] for value in values)
        group["top"] = min(value[5] for value in values)
        group["bottom"] = max(value[7] for value in values)
        group["left"] = min(value[4] for value in values)
        group["right"] = max(value[6] for value in values)
        group["row_gaps"] = [
            later["center_y"] - earlier["center_y"]
            for earlier, later in zip(group["rows"], group["rows"][1:])
        ]

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    @classmethod
    def _geometrically_adjacent(
        cls,
        first: tuple[float, float, float, float, float, float, float, float],
        second: tuple[float, float, float, float, float, float, float, float],
    ) -> bool:
        """Allow horizontal, vertical, and grid-neighbor connections."""
        width_a, height_a, center_x_a, center_y_a, left_a, top_a, right_a, bottom_a = first
        width_b, height_b, center_x_b, center_y_b, left_b, top_b, right_b, bottom_b = second
        if min(width_a, width_b, height_a, height_b) <= 0.0:
            return False

        vertical_overlap = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
        horizontal_overlap = max(0.0, min(right_a, right_b) - max(left_a, left_b))
        horizontal = (
            vertical_overlap / min(height_a, height_b) >= cls.PEER_MIN_VERTICAL_OVERLAP_RATIO
            and max(left_a, left_b) - min(right_a, right_b)
            <= cls.PEER_MAX_HORIZONTAL_GAP_RATIO * max(width_a, width_b)
        )
        vertical = (
            horizontal_overlap / min(width_a, width_b) >= cls.PEER_MIN_HORIZONTAL_OVERLAP_RATIO
            and max(top_a, top_b) - min(bottom_a, bottom_b)
            <= cls.PEER_MAX_VERTICAL_GAP_RATIO * max(height_a, height_b)
        )
        # Handle columns whose boxes do not quite overlap horizontally but are
        # still clearly adjacent by center alignment.
        aligned_vertical = (
            abs(center_x_a - center_x_b) <= 1.5 * max(width_a, width_b)
            and max(top_a, top_b) - min(bottom_a, bottom_b)
            <= cls.PEER_MAX_VERTICAL_GAP_RATIO * max(height_a, height_b)
        )
        return horizontal or vertical or aligned_vertical

    @classmethod
    def _spatial_order_indices(
        cls, candidates: list[dict[str, Any]], indices: list[int]
    ) -> list[int]:
        """Order candidate presentation by horizontal rows, then left-to-right."""
        geometries = {
            index: cls._candidate_geometry(candidates[index])
            for index in indices
            if 0 <= index < len(candidates)
        }
        valid = [index for index in indices if geometries.get(index) is not None]
        invalid = [index for index in indices if geometries.get(index) is None]
        rows: list[dict[str, Any]] = []

        for index in sorted(valid, key=lambda item: (geometries[item][3], item)):
            geometry = geometries[index]
            assert geometry is not None
            row = next(
                (
                    candidate_row
                    for candidate_row in rows
                    if any(
                        cls._same_horizontal_row(
                            geometry,
                            geometries[row_index],
                        )
                        for row_index in candidate_row["indices"]
                    )
                ),
                None,
            )
            if row is None:
                rows.append({"center_y": geometry[3], "indices": [index]})
            else:
                row["indices"].append(index)
                row["center_y"] = sum(
                    geometries[row_index][3] for row_index in row["indices"]
                ) / len(row["indices"])

        ordered: list[int] = []
        for row in sorted(rows, key=lambda candidate_row: candidate_row["center_y"]):
            ordered.extend(
                sorted(row["indices"], key=lambda index: (geometries[index][2], index))
            )
        return ordered + invalid

    @classmethod
    def _same_horizontal_row(
        cls,
        first: tuple[float, float, float, float, float, float, float, float],
        second: tuple[float, float, float, float, float, float, float, float],
    ) -> bool:
        height_a, center_y_a = first[1], first[3]
        height_b, center_y_b = second[1], second[3]
        vertical_overlap = max(
            0.0,
            min(first[7], second[7]) - max(first[5], second[5]),
        )
        return (
            vertical_overlap / min(height_a, height_b) >= cls.PEER_MIN_VERTICAL_OVERLAP_RATIO
            or abs(center_y_a - center_y_b)
            <= cls.PEER_MAX_CENTER_Y_DISTANCE_RATIO * max(height_a, height_b)
        )

    @staticmethod
    def _candidate_geometry(
        candidate: Any,
    ) -> tuple[float, float, float, float, float, float, float, float] | None:
        if not isinstance(candidate, dict):
            return None
        bbox = candidate.get("bbox_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            left, top, right, bottom = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        width = right - left
        height = bottom - top
        if width <= 0.0 or height <= 0.0:
            return None
        return width, height, (left + right) / 2.0, (top + bottom) / 2.0, left, top, right, bottom

    @classmethod
    def _geometrically_compatible(
        cls,
        first: tuple[float, float, float, float, float, float, float, float],
        second: tuple[float, float, float, float, float, float, float, float],
    ) -> bool:
        width_a, height_a, center_x_a, center_y_a, left_a, top_a, right_a, bottom_a = first
        width_b, height_b, center_x_b, center_y_b, left_b, top_b, right_b, bottom_b = second
        if min(height_a, height_b) / max(height_a, height_b) < cls.PEER_MIN_HEIGHT_RATIO:
            return False

        overlap = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
        vertical_overlap_ratio = overlap / min(height_a, height_b)
        similar_center_y = abs(center_y_a - center_y_b) <= (
            cls.PEER_MAX_CENTER_Y_DISTANCE_RATIO * max(height_a, height_b)
        )
        if vertical_overlap_ratio < cls.PEER_MIN_VERTICAL_OVERLAP_RATIO and not similar_center_y:
            return False

        center_x_distance = abs(center_x_a - center_x_b)
        if center_x_distance < cls.PEER_MIN_CENTER_X_DISTANCE_RATIO * min(width_a, width_b):
            return False
        horizontal_gap = max(left_a, left_b) - min(right_a, right_b)
        return horizontal_gap <= cls.PEER_MAX_HORIZONTAL_GAP_RATIO * max(width_a, width_b)

    @classmethod
    def _prepare_focus_image(
        cls,
        image: Image.Image,
        candidates: list[dict[str, Any]],
        candidate_indices: list[int] | None = None,
        focus_group_ids: list[int] | None = None,
        use_montage: bool = False,
    ) -> tuple[
        Image.Image,
        Image.Image,
        tuple[int, int, int, int] | None,
        bool,
        list[int],
        str,
        list[int],
        list[int],
        list[int],
        list[list[int]],
        dict[int, list[int]],
    ]:
        """Make one geometrically selected, annotated image for focus inference."""
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise StateObservationError("focus resolver image must be a non-empty PIL image")

        image_width, image_height = image.size
        candidate_boxes: list[tuple[int, int, int, int, int, int]] = []
        for position, candidate in enumerate(candidates):
            index = candidate_indices[position] if candidate_indices is not None else position
            group_id = focus_group_ids[position] if focus_group_ids is not None else 0
            bbox = candidate.get("bbox_norm") if isinstance(candidate, dict) else None
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                left, top, right, bottom = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            left, right = sorted((left, right))
            top, bottom = sorted((top, bottom))
            left = max(0.0, min(1000.0, left))
            right = max(0.0, min(1000.0, right))
            top = max(0.0, min(1000.0, top))
            bottom = max(0.0, min(1000.0, bottom))
            pixel_box = (
                max(0, min(image_width - 1, round(left * image_width / 1000))),
                max(0, min(image_height - 1, round(top * image_height / 1000))),
                max(1, min(image_width, round(right * image_width / 1000))),
                max(1, min(image_height, round(bottom * image_height / 1000))),
            )
            if pixel_box[2] > pixel_box[0] and pixel_box[3] > pixel_box[1]:
                candidate_boxes.append((*pixel_box, index, group_id))

        if use_montage:
            montage = cls._prepare_candidate_montage(image, candidate_boxes)
            if montage is not None:
                return montage

        # A single box does not provide the neighboring peer context the
        # resolver needs. The full annotated frame is the safe fallback.
        roi_used = len(candidate_boxes) >= 2
        if roi_used:
            left = min(box[0] for box in candidate_boxes)
            top = min(box[1] for box in candidate_boxes)
            right = max(box[2] for box in candidate_boxes)
            bottom = max(box[3] for box in candidate_boxes)
            margin_x = max(8, round((right - left) * cls.ROI_MARGIN_FRACTION))
            margin_y = max(8, round((bottom - top) * cls.ROI_MARGIN_FRACTION))
            roi = (
                max(0, left - margin_x),
                max(0, top - margin_y),
                min(image_width, right + margin_x),
                min(image_height, bottom + margin_y),
            )
            roi_area_fraction = ((roi[2] - roi[0]) * (roi[3] - roi[1])) / (image_width * image_height)
            if roi_area_fraction > cls.ROI_MAX_AREA_FRACTION:
                roi_used = False
        else:
            roi = (0, 0, image_width, image_height)

        if not roi_used:
            roi = (0, 0, image_width, image_height)

        crop = image.crop(roi).copy()
        unannotated_crop = crop.copy()
        draw = ImageDraw.Draw(crop)
        annotated_indices: list[int] = []
        prepared_candidate_bboxes: dict[int, list[int]] = {}
        for left, top, right, bottom, index, _group_id in candidate_boxes:
            if right <= roi[0] or left >= roi[2] or bottom <= roi[1] or top >= roi[3]:
                continue
            x1 = max(left, roi[0]) - roi[0]
            y1 = max(top, roi[1]) - roi[1]
            cls._draw_focus_index_label(draw, index, x1, y1, crop.size)
            annotated_indices.append(index)
            prepared_candidate_bboxes[index] = [
                max(0, left - roi[0]),
                max(0, top - roi[1]),
                min(crop.width, right - roi[0]),
                min(crop.height, bottom - roi[1]),
            ]

        input_image = crop
        unannotated_input_image = unannotated_crop
        crop_area_fraction = (crop.width * crop.height) / (image_width * image_height)
        if roi_used and crop_area_fraction < cls.ROI_MAX_AREA_FRACTION:
            scale = min(
                cls.ENLARGE_FACTOR,
                cls.MAX_ENLARGED_DIMENSION / crop.width,
                cls.MAX_ENLARGED_DIMENSION / crop.height,
            )
            if scale > 1.0:
                input_image = crop.resize(
                    (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                unannotated_input_image = unannotated_crop.resize(
                    input_image.size, Image.Resampling.LANCZOS
                )
                prepared_candidate_bboxes = {
                    index: [round(value * scale) for value in box]
                    for index, box in prepared_candidate_bboxes.items()
                }
        return (
            input_image,
            unannotated_input_image,
            roi,
            roi_used,
            annotated_indices,
            "roi" if roi_used else "full_image",
            [],
            [0, 0],
            [0, 0],
            [],
            prepared_candidate_bboxes,
        )

    @staticmethod
    def _focus_label_font(image_size: tuple[int, int]) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_size = max(16, min(30, round(min(image_size) * 0.035)))
        for font_path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                continue
        return ImageFont.load_default()

    @classmethod
    def _draw_focus_index_label(
        cls,
        draw: ImageDraw.ImageDraw,
        index: int,
        anchor_x: int,
        anchor_y: int,
        image_size: tuple[int, int],
    ) -> None:
        label = str(index)
        font = cls._focus_label_font(image_size)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        pad_x = max(5, round(text_height * 0.35))
        pad_y = max(3, round(text_height * 0.20))
        badge_width = text_width + 2 * pad_x
        badge_height = text_height + 2 * pad_y
        image_width, image_height = image_size
        label_x = min(max(2, anchor_x), max(2, image_width - badge_width - 2))
        if anchor_y >= badge_height + 5:
            label_y = anchor_y - badge_height - 4
        else:
            label_y = anchor_y + 3
        label_y = min(max(2, label_y), max(2, image_height - badge_height - 2))
        draw.rounded_rectangle(
            (label_x, label_y, label_x + badge_width, label_y + badge_height),
            radius=max(3, round(badge_height * 0.18)),
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=1,
        )
        draw.text(
            (label_x + pad_x, label_y + pad_y - text_box[1]),
            label,
            font=font,
            fill=(255, 255, 255),
        )

    @classmethod
    def _prepare_candidate_montage(
        cls,
        image: Image.Image,
        candidate_boxes: list[tuple[int, int, int, int, int, int]],
    ) -> tuple[
        Image.Image,
        Image.Image,
        tuple[int, int, int, int] | None,
        bool,
        list[int],
        str,
        list[int],
        list[int],
        list[int],
        list[list[int]],
        dict[int, list[int]],
    ] | None:
        """Build one context-preserving crop tile for each candidate group."""
        if not candidate_boxes:
            return None

        try:
            montage_max_dimension = 2048
            gap = 24
            padding = 12
            group_tiles: list[dict[str, Any]] = []
            for group_id, group_boxes in cls._group_candidate_boxes(candidate_boxes):
                left = min(box[0] for box in group_boxes)
                top = min(box[1] for box in group_boxes)
                right = max(box[2] for box in group_boxes)
                bottom = max(box[3] for box in group_boxes)
                margin_x = max(12, round((right - left) * 0.10))
                margin_y = max(12, round((bottom - top) * 0.20))
                crop_left = max(0, left - margin_x)
                crop_top = max(0, top - margin_y)
                crop_right = min(image.width, right + margin_x)
                crop_bottom = min(image.height, bottom + margin_y)
                crop = image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
                if crop.width <= 0 or crop.height <= 0:
                    return None
                group_tiles.append({
                    "group_id": group_id,
                    "indices": [box[4] for box in group_boxes],
                    "boxes": group_boxes,
                    "crop": crop,
                    "crop_left": crop_left,
                    "crop_top": crop_top,
                })
            if not group_tiles:
                return None

            columns = min(3, max(1, round(len(group_tiles) ** 0.5)))
            rows = (len(group_tiles) + columns - 1) // columns
            layout_rows = [
                group_tiles[row * columns:(row + 1) * columns]
                for row in range(rows)
            ]
            natural_widths = [
                sum(tile["crop"].width for tile in row) + gap * (len(row) - 1)
                for row in layout_rows
            ]
            natural_heights = [max(tile["crop"].height for tile in row) for row in layout_rows]
            natural_width = max(natural_widths) + 2 * padding
            natural_height = sum(natural_heights) + gap * (rows - 1) + 2 * padding
            scale = min(
                2.0,
                montage_max_dimension / natural_width,
                montage_max_dimension / natural_height,
            )
            if scale <= 0.0:
                return None

            scaled_sizes = [
                (max(1, round(tile["crop"].width * scale)), max(1, round(tile["crop"].height * scale)))
                for tile in group_tiles
            ]
            row_heights = [
                max(scaled_sizes[index][1] for index in range(row * columns, min((row + 1) * columns, len(group_tiles))))
                for row in range(rows)
            ]
            row_widths = [
                sum(scaled_sizes[index][0] for index in range(row * columns, min((row + 1) * columns, len(group_tiles))))
                + gap * (len(layout_rows[row]) - 1)
                for row in range(rows)
            ]
            montage_width = max(row_widths) + 2 * padding
            montage_height = sum(row_heights) + gap * (rows - 1) + 2 * padding
            if montage_width > montage_max_dimension or montage_height > montage_max_dimension:
                return None

            montage = Image.new("RGB", (montage_width, montage_height), (32, 32, 32))
            unannotated_montage = Image.new(
                "RGB", (montage_width, montage_height), (32, 32, 32)
            )
            annotated_indices: list[int] = []
            tile_sizes: list[list[int]] = []
            prepared_candidate_bboxes: dict[int, list[int]] = {}
            draw = ImageDraw.Draw(montage)
            y = padding
            for row in range(rows):
                x = padding
                row_start = row * columns
                row_end = min((row + 1) * columns, len(group_tiles))
                for tile_index in range(row_start, row_end):
                    tile_data = group_tiles[tile_index]
                    tile_width, tile_height = scaled_sizes[tile_index]
                    tile = tile_data["crop"].resize(
                        (tile_width, tile_height), Image.Resampling.LANCZOS
                    )
                    unannotated_tile = tile.copy()
                    tile_draw = ImageDraw.Draw(tile)
                    scale_x = tile_width / tile_data["crop"].width
                    scale_y = tile_height / tile_data["crop"].height
                    for left, top, right, bottom, index, _ in tile_data["boxes"]:
                        label_x = round((left - tile_data["crop_left"]) * scale_x)
                        label_y = round((top - tile_data["crop_top"]) * scale_y)
                        cls._draw_focus_index_label(
                            tile_draw,
                            index,
                            label_x,
                            label_y,
                            tile.size,
                        )
                        annotated_indices.append(index)
                        prepared_candidate_bboxes[index] = [
                            x + label_x,
                            y + label_y,
                            x + round((right - tile_data["crop_left"]) * scale_x),
                            y + round((bottom - tile_data["crop_top"]) * scale_y),
                        ]
                    unannotated_montage.paste(unannotated_tile, (x, y))
                    montage.paste(tile, (x, y))
                    tile_sizes.append([tile_width, tile_height])
                    x += tile_width + gap
                y += row_heights[row] + gap

            return (
                montage,
                unannotated_montage,
                None,
                False,
                annotated_indices,
                "group_montage",
                annotated_indices,
                [rows, columns],
                [montage.width, montage.height],
                tile_sizes,
                prepared_candidate_bboxes,
            )
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _group_candidate_boxes(
        candidate_boxes: list[tuple[int, int, int, int, int, int]],
    ) -> list[tuple[int, list[tuple[int, int, int, int, int, int]]]]:
        grouped: list[tuple[int, list[tuple[int, int, int, int, int, int]]]] = []
        for box in candidate_boxes:
            if not grouped or grouped[-1][0] != box[5]:
                grouped.append((box[5], []))
            grouped[-1][1].append(box)
        return grouped

    @staticmethod
    def _parse_output(raw_output: str, candidate_count: int) -> int | None:
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise StateObservationError("Focus resolver output is empty")
        try:
            payload = json.loads(raw_output.strip())
        except (TypeError, json.JSONDecodeError) as exc:
            raise StateObservationError(f"Focus resolver output is malformed JSON: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"focused_index"}:
            raise StateObservationError(
                "Focus resolver output must be exactly {\"focused_index\": integer or null}"
            )
        focused_index = payload["focused_index"]
        if focused_index is None:
            return None
        if isinstance(focused_index, bool) or not isinstance(focused_index, int):
            raise StateObservationError("Focus resolver focused_index must be an integer or null")
        if not 0 <= focused_index < candidate_count:
            raise StateObservationError(
                f"Focus resolver focused_index {focused_index} is outside candidate range"
            )
        return focused_index


__all__ = ["FOCUS_RESOLVER_PROMPT_TEMPLATE", "FocusResolver"]
