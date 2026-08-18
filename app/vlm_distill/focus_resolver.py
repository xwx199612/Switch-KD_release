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
    DEBUG_CV_PREPARED_IMAGE_PATH = f"{tempfile.gettempdir()}/focus_resolver_cv_prepared.jpg"
    DEBUG_CV_PREPARED_DEBUG_IMAGE_PATH = f"{tempfile.gettempdir()}/focus_resolver_cv_prepared_debug.jpg"
    DEBUG_CV_PREPARED_METADATA_PATH = f"{tempfile.gettempdir()}/focus_resolver_cv_prepared.json"
    DEBUG_CV_FINAL_IMAGE_PATH = f"{tempfile.gettempdir()}/focus_resolver_cv_final.jpg"
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
    ENLARGEMENT_OBS_BOUNDARY_TOLERANCE_PX = 2
    ENLARGEMENT_EDGE_SAMPLE_INSET = 0.15
    ENLARGEMENT_BOUNDARY_MIN_SCORE = 0.55
    ENLARGEMENT_BOUNDARY_MIN_SAMPLE_SUPPORT = 0.60
    ENLARGEMENT_CONTINUATION_MIN_SCORE = 0.55
    ENLARGEMENT_CONTINUATION_MIN_TAIL_STEPS = 2
    ENLARGEMENT_CONTINUATION_MIN_PATH_SUPPORT = 0.60
    ENLARGEMENT_CONTINUATION_MIN_SAMPLE_SUPPORT = 0.60
    ENLARGEMENT_CENSORED_CONFIDENCE_FACTOR = 0.70
    ENLARGEMENT_EDGE_COHERENCE_GRADIENT_RADIUS = 2
    ENLARGEMENT_EDGE_COHERENCE_MIN_SAMPLE_STRENGTH = 0.35
    ENLARGEMENT_EDGE_COHERENCE_MIN_SPAN_SUPPORT = 0.55
    ENLARGEMENT_EDGE_COHERENCE_MIN_SCORE = 0.55
    ENLARGEMENT_EDGE_COHERENCE_LOCAL_RADIUS_STEPS = 1
    ENLARGEMENT_EDGE_COHERENCE_LUMINANCE_NORMALIZER = 0.20
    ENLARGEMENT_EDGE_COHERENCE_COLOR_NORMALIZER = 0.35
    ENLARGEMENT_STRUCTURE_MIN_SAMPLE_STRENGTH = 0.12
    DEVICE_BOUNDARY_MIN_SPAN_FRACTION = 0.70
    DEVICE_BOUNDARY_SEARCH_FRACTION = 0.25
    DEVICE_BOUNDARY_MIN_SCORE = 0.60
    DEVICE_BOUNDARY_GRADIENT_RADIUS = 2
    DEVICE_BOUNDARY_MIN_SAMPLE_STRENGTH = 0.35
    ENLARGEMENT_DEVICE_BOUNDARY_TOLERANCE_PX = 4.0
    ENLARGEMENT_DEVICE_BOUNDARY_TOLERANCE_RATIO = 0.01
    NATURAL_BASELINE_MIN_PEERS = 1
    NATURAL_BASELINE_MAX_PADDING_RATIO_X = 0.30
    NATURAL_BASELINE_MAX_PADDING_RATIO_Y = 0.30
    NATURAL_CONTAINER_MIN_EDGE_SCORE = 0.55
    NATURAL_CONTAINER_MIN_SPAN_SUPPORT = 0.55
    NATURAL_CONTAINER_MIN_ORIENTED_STRENGTH = 0.35
    NATURAL_CONTAINER_LOCAL_RIDGE_RADIUS_STEPS = 1
    NATURAL_CONTAINER_DENSE_SCAN_RADIUS_PX = 8
    NATURAL_CONTAINER_LOCAL_RIDGE_TOLERANCE = 0.03
    ENLARGEMENT_COMPLETION_MIN_RETAINED_RATIO = 0.70
    ENLARGEMENT_MIRRORED_CONFIDENCE_FACTOR = 0.75
    ENLARGEMENT_CARD_CELL_OUTER_X = 0.30
    ENLARGEMENT_CARD_CELL_OUTER_Y = 0.30
    ENLARGEMENT_CARD_OBS_MARGIN_X = 0.20
    ENLARGEMENT_CARD_OBS_MARGIN_Y = 0.25
    ENLARGEMENT_SIBLING_PROTECTED_CORE_X = 0.60
    ENLARGEMENT_SIBLING_PROTECTED_CORE_Y = 0.60
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
        source_device_geometry = self._estimate_source_device_viewport(image)
        prepared_montage_tiles_by_index = {
            int(index): list(tile["bbox"])
            for tile in montage_tile_sizes
            if isinstance(tile, dict)
            and isinstance(tile.get("bbox"), (list, tuple))
            for index in tile.get("candidate_indices", [])
            if isinstance(index, int)
        }
        prepared_device_geometry_by_index = (
            self._map_source_device_viewport_to_prepared(
                source_device_geometry,
                image.size,
                unannotated_image.size,
                focus_image_mode,
                roi_bbox,
                montage_tile_sizes,
            )
        )
        peer_analysis = self._build_peer_analysis(
            candidate_groups, prepared_candidate_bboxes
        )
        try:
            unannotated_image.convert("RGB").save(
                self.DEBUG_CV_PREPARED_IMAGE_PATH,
                format="JPEG",
                quality=95,
            )
        except (OSError, ValueError):
            pass

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
            source_device_geometry,
            prepared_device_geometry_by_index,
            prepared_montage_tiles_by_index,
        )
        cv_debug_paths: dict[str, str | None] = {
            "focus_cv_prepared_image_path": self.DEBUG_CV_PREPARED_IMAGE_PATH,
            "focus_cv_prepared_debug_image_path": None,
            "focus_cv_prepared_metadata_path": None,
            "focus_cv_final_image_path": None,
        }
        try:
            self._save_cv_prepared_debug_artifacts(
                unannotated_image,
                candidates,
                prepared_candidate_bboxes,
                candidate_groups,
                peer_analysis,
                enlargement_sibling_analysis,
                visual_evidence,
                focus_image_mode,
                montage_grid,
                montage_size,
                montage_tile_sizes,
                roi_bbox,
                source_device_geometry,
            )
            cv_debug_paths["focus_cv_prepared_debug_image_path"] = self.DEBUG_CV_PREPARED_DEBUG_IMAGE_PATH
            cv_debug_paths["focus_cv_prepared_metadata_path"] = self.DEBUG_CV_PREPARED_METADATA_PATH
        except (OSError, ValueError, TypeError):
            pass
        v5_cascade = self._run_visual_focus_cascade(
            visual_evidence,
            peer_analysis["peer_sets"],
            peer_analysis["peer_group_by_index"],
            peer_analysis["isolated_indices"],
            enlargement_peer_sets=enlargement_sibling_analysis["sibling_sets"],
        )
        try:
            self._save_cv_final_debug_image(
                unannotated_image,
                candidates,
                prepared_candidate_bboxes,
                visual_evidence,
                v5_cascade,
            )
            cv_debug_paths["focus_cv_final_image_path"] = self.DEBUG_CV_FINAL_IMAGE_PATH
        except (OSError, ValueError, TypeError):
            pass
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
            "source_device_viewport_bbox": source_device_geometry.get("source_device_viewport_bbox"),
            "source_device_viewport_valid": source_device_geometry.get("source_device_viewport_valid"),
            "source_device_viewport_confidence": source_device_geometry.get("source_device_viewport_confidence"),
            "focus_montage_group_tile_count": len(candidate_groups)
            if focus_image_mode == "group_montage" else 0,
            "focus_montage_group_tile_indices": candidate_groups
            if focus_image_mode == "group_montage" else [],
            "focus_debug_image_path": focus_debug_image_path,
            "focus_debug_unannotated_image_path": focus_debug_unannotated_image_path,
            "focus_visual_evidence_space": "prepared",
            "focus_peer_debug_image_path": focus_peer_debug_image_path,
            **cv_debug_paths,
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
        source_device_geometry: dict[str, Any] | None = None,
        prepared_device_geometry_by_index: dict[int, dict[str, Any]] | None = None,
        prepared_montage_tiles_by_index: dict[int, list[int]] | None = None,
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
            if source_device_geometry:
                item.update(source_device_geometry)
            prepared_device_geometry = (
                (prepared_device_geometry_by_index or {}).get(
                    index,
                    (prepared_device_geometry_by_index or {}).get(-1, {}),
                )
            )
            if prepared_device_geometry:
                item.update(prepared_device_geometry)
            prepared_tile = (prepared_montage_tiles_by_index or {}).get(index)
            if prepared_tile is not None:
                item["prepared_montage_tile_bbox"] = prepared_tile

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

        def measure_direct_extent(
            item: dict[str, Any],
            observation_override: tuple[float, float, float, float] | None = None,
        ) -> None:
            semantic = clipped_box(item.get("prepared_bbox"))
            cell = clipped_box(item.get("visual_cell_bbox"))
            if semantic is None or cell is None:
                item["enlargement_extent_reason"] = "invalid_geometry"
                item["extent_reliable"] = False
                return
            left, top, right, bottom = semantic
            width = right - left
            height = bottom - top
            legacy_observation = clipped_box((
                max(cell[0], left - width * cls.ENLARGEMENT_LOCAL_MARGIN_X),
                max(cell[1], top - height * cls.ENLARGEMENT_LOCAL_MARGIN_Y),
                min(cell[2], right + width * cls.ENLARGEMENT_LOCAL_MARGIN_X),
                min(cell[3], bottom + height * cls.ENLARGEMENT_LOCAL_MARGIN_Y),
            )) or semantic
            item["enlargement_local_observation_bbox"] = [
                round(value, 2) for value in legacy_observation
            ]
            observation = clipped_box(observation_override) if observation_override is not None else legacy_observation
            if observation is None or not (
                observation[0] <= left and observation[1] <= top
                and observation[2] >= right and observation[3] >= bottom
            ):
                item["enlargement_extent_reason"] = "invalid_geometry"
                item["extent_reliable"] = False
                return

            def luminance(color: tuple[int, int, int]) -> float:
                return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255.0

            step = max(1, min(4, int(round(min(width, height) * 0.025))))
            growth: dict[str, int] = {"left": 0, "right": 0, "top": 0, "bottom": 0}
            boundary_debug: dict[str, dict[str, Any]] = {}
            sample_count = max(5, min(15, int(round(max(width, height) / 20.0))))

            def coordinate(side: str, distance: int, fraction: float, farther: int = 0) -> tuple[int, int]:
                if side in ("left", "right"):
                    y = int(round(top + fraction * height))
                    x = int(round(left - distance - farther if side == "left" else right + distance + farther))
                else:
                    x = int(round(left + fraction * width))
                    y = int(round(top - distance - farther if side == "top" else bottom + distance + farther))
                return (
                    max(int(observation[0]), min(int(observation[2]) - 1, x)),
                    max(int(observation[1]), min(int(observation[3]) - 1, y)),
                )

            def inner_coordinate(side: str, fraction: float) -> tuple[int, int]:
                inset = max(1, step)
                if side in ("left", "right"):
                    y = int(round(top + fraction * height))
                    x = int(round(left + inset if side == "left" else right - inset))
                else:
                    x = int(round(left + fraction * width))
                    y = int(round(top + inset if side == "top" else bottom - inset))
                return (
                    max(int(observation[0]), min(int(observation[2]) - 1, x)),
                    max(int(observation[1]), min(int(observation[3]) - 1, y)),
                )

            def pixel_at(x: int, y: int) -> tuple[int, int, int]:
                return pixel_data[
                    max(int(observation[0]), min(int(observation[2]) - 1, x)),
                    max(int(observation[1]), min(int(observation[3]) - 1, y)),
                ]

            def gradient_strength(
                first: tuple[int, int, int],
                second: tuple[int, int, int],
            ) -> float:
                luminance_gradient = abs(luminance(first) - luminance(second))
                color_gradient = math.sqrt(sum(
                    ((first[channel] - second[channel]) / 255.0) ** 2
                    for channel in range(3)
                ) / 3.0)
                return cls._clamp01(
                    0.45 * cls._clamp01(
                        luminance_gradient
                        / cls.ENLARGEMENT_EDGE_COHERENCE_LUMINANCE_NORMALIZER
                    )
                    + 0.55 * cls._clamp01(
                        color_gradient
                        / cls.ENLARGEMENT_EDGE_COHERENCE_COLOR_NORMALIZER
                    )
                )

            def edge_profile(
                side: str,
                distance: int,
            ) -> dict[str, float]:
                radius = cls.ENLARGEMENT_EDGE_COHERENCE_GRADIENT_RADIUS
                strengths: list[float] = []
                orientations: list[float] = []
                structure_values: list[float] = []
                background_values: list[float] = []
                inset = cls.ENLARGEMENT_EDGE_SAMPLE_INSET
                for sample_index in range(sample_count):
                    fraction = inset + (1.0 - 2.0 * inset) * (sample_index + 0.5) / sample_count
                    x, y = coordinate(side, distance, fraction)
                    if side == "top":
                        inside, outside, farther = (
                            pixel_at(x, y + radius),
                            pixel_at(x, y - radius),
                            pixel_at(x, y - 2 * radius),
                        )
                        tangent_first, tangent_second = (
                            pixel_at(x - radius, y), pixel_at(x + radius, y)
                        )
                    elif side == "bottom":
                        inside, outside, farther = (
                            pixel_at(x, y - radius),
                            pixel_at(x, y + radius),
                            pixel_at(x, y + 2 * radius),
                        )
                        tangent_first, tangent_second = (
                            pixel_at(x - radius, y), pixel_at(x + radius, y)
                        )
                    elif side == "left":
                        inside, outside, farther = (
                            pixel_at(x + radius, y),
                            pixel_at(x - radius, y),
                            pixel_at(x - 2 * radius, y),
                        )
                        tangent_first, tangent_second = (
                            pixel_at(x, y - radius), pixel_at(x, y + radius)
                        )
                    else:
                        inside, outside, farther = (
                            pixel_at(x - radius, y),
                            pixel_at(x + radius, y),
                            pixel_at(x + 2 * radius, y),
                        )
                        tangent_first, tangent_second = (
                            pixel_at(x, y - radius), pixel_at(x, y + radius)
                        )
                    normal_strength = gradient_strength(inside, outside)
                    tangential_strength = gradient_strength(
                        tangent_first,
                        tangent_second,
                    )
                    orientation = normal_strength / max(
                        normal_strength + tangential_strength,
                        1e-6,
                    )
                    strengths.append(normal_strength * orientation)
                    orientations.append(orientation)
                    structure_values.append(max(normal_strength, tangential_strength))
                    background_values.append(cls._clamp01(
                        1.0 - abs(luminance(outside) - luminance(farther)) / 0.08
                    ))
                span_support = sum(
                    value >= cls.ENLARGEMENT_EDGE_COHERENCE_MIN_SAMPLE_STRENGTH
                    for value in strengths
                ) / max(len(strengths), 1)
                mean_strength = sum(strengths) / max(len(strengths), 1)
                return {
                    "edge_coherence_score": cls._clamp01(
                        0.65 * span_support + 0.35 * mean_strength
                    ),
                    "edge_span_support": span_support,
                    "edge_mean_strength": mean_strength,
                    "edge_orientation_score": sum(orientations) / max(len(orientations), 1),
                    "outer_background_contrast": mean_strength,
                    "background_stability": sum(background_values) / max(len(background_values), 1),
                    "structure_persistence": sum(
                        value >= cls.ENLARGEMENT_STRUCTURE_MIN_SAMPLE_STRENGTH
                        for value in structure_values
                    ) / max(len(structure_values), 1),
                }

            for side in growth:
                maximum = int(round(
                    (left - observation[0]) if side == "left"
                    else (observation[2] - right) if side == "right"
                    else (top - observation[1]) if side == "top"
                    else (observation[3] - bottom)
                ))
                if maximum < step:
                    boundary_debug[side] = {
                        "found": False, "growth": 0, "score": 0.0,
                        "confidence": 0.0, "reason": "insufficient_scan_space",
                        "obs_hit": False, "state": "unresolved",
                        "continuation_score": 0.0,
                        "continuation_confidence": 0.0,
                        "continuation_path_support": 0.0,
                        "continuation_tail_support": 0.0,
                        "continuation_sample_support": 0.0,
                        "censored": False,
                        "sample_count": 0,
                        "boundary_sample_count": 0,
                        "continuation_sample_count": 0,
                        "unresolved_sample_count": 0,
                        "edge_coherence_score": 0.0,
                        "edge_span_support": 0.0,
                        "edge_mean_strength": 0.0,
                        "edge_orientation_score": 0.0,
                        "edge_distance": None,
                        "edge_found": False,
                        "structure_persistence": 0.0,
                        "background_takeover_score": 0.0,
                    }
                    continue
                distances = list(range(step, maximum + 1, step))
                profiles = {
                    distance: edge_profile(side, distance)
                    for distance in distances
                }
                coherent_boundary: tuple[int, dict[str, float]] | None = None
                for position, distance in enumerate(distances):
                    profile = profiles[distance]
                    nearby = [
                        profiles[distances[neighbor]]["edge_coherence_score"]
                        for neighbor in range(
                            max(0, position - cls.ENLARGEMENT_EDGE_COHERENCE_LOCAL_RADIUS_STEPS),
                            min(
                                len(distances),
                                position
                                + cls.ENLARGEMENT_EDGE_COHERENCE_LOCAL_RADIUS_STEPS
                                + 1,
                            ),
                        )
                    ]
                    if (
                        profile["edge_coherence_score"]
                        >= cls.ENLARGEMENT_EDGE_COHERENCE_MIN_SCORE
                        and profile["edge_span_support"]
                        >= cls.ENLARGEMENT_EDGE_COHERENCE_MIN_SPAN_SUPPORT
                        and profile["background_stability"] >= 0.55
                        and profile["edge_coherence_score"] >= max(nearby)
                    ):
                        coherent_boundary = (distance, profile)
                        break
                positions: list[tuple[int, float]] = []
                scores: list[float] = []
                inset = cls.ENLARGEMENT_EDGE_SAMPLE_INSET
                for sample_index in range(sample_count):
                    fraction = inset + (1.0 - 2.0 * inset) * (sample_index + 0.5) / sample_count
                    inner = luminance(pixel_data[inner_coordinate(side, fraction)])
                    found_position: tuple[int, float] | None = None
                    for distance in range(step, maximum + 1, step):
                        current = luminance(pixel_data[coordinate(side, distance, fraction)])
                        farther = luminance(pixel_data[coordinate(side, distance, fraction, step)])
                        farther_next = luminance(pixel_data[coordinate(side, distance, fraction, 2 * step)])
                        inner_support = cls._clamp01(1.0 - abs(current - inner) / 0.18)
                        outward_change = cls._clamp01(abs(current - farther) / 0.12)
                        boundary_gradient = outward_change
                        background_stability = cls._clamp01(1.0 - abs(farther - farther_next) / 0.08)
                        score = (
                            0.35 * outward_change
                            + 0.30 * boundary_gradient
                            + 0.20 * background_stability
                            + 0.15 * inner_support
                        )
                        if (
                            score >= cls.ENLARGEMENT_BOUNDARY_MIN_SCORE
                            and background_stability >= 0.55
                            and inner_support >= 0.30
                        ):
                            found_position = (distance, score)
                            break
                    if found_position is not None:
                        positions.append(found_position)
                        scores.append(found_position[1])
                required = max(1, math.ceil(sample_count * cls.ENLARGEMENT_BOUNDARY_MIN_SAMPLE_SUPPORT))
                selected_profile = (
                    coherent_boundary[1]
                    if coherent_boundary is not None
                    else max(
                        profiles.values(),
                        key=lambda profile: profile["edge_coherence_score"],
                    )
                )
                continuation_values: list[float] = []
                for profile in profiles.values():
                    background_takeover = cls._clamp01(
                        profile["background_stability"]
                        * (1.0 - profile["structure_persistence"])
                    )
                    continuation_values.append(cls._clamp01(
                        0.50 * profile["structure_persistence"]
                        + 0.30 * (1.0 - background_takeover)
                        + 0.20 * (1.0 - profile["edge_coherence_score"])
                    ))
                continuation_path_support = sum(
                    value >= cls.ENLARGEMENT_CONTINUATION_MIN_SCORE
                    for value in continuation_values
                ) / max(len(continuation_values), 1)
                tail_steps = min(
                    cls.ENLARGEMENT_CONTINUATION_MIN_TAIL_STEPS,
                    len(continuation_values),
                )
                continuation_tail_support = (
                    sum(continuation_values[-tail_steps:]) / tail_steps
                    if tail_steps else 0.0
                )
                continuation_tail_structure_support = (
                    sum(
                        profiles[distance]["structure_persistence"]
                        for distance in distances[-tail_steps:]
                    ) / tail_steps
                    if tail_steps else 0.0
                )
                continuation_score = (
                    sum(continuation_values) / len(continuation_values)
                    if continuation_values else 0.0
                )
                continuation_allowed = (
                    observation_override is not None
                    and item.get("enlargement_extent_observation_source")
                    == "pure_card_v5_3"
                    and bool(item.get("enlargement_card_observation_valid"))
                    and not bool(item.get(
                        "enlargement_card_observation_intersects_other_sibling_core"
                    ))
                )
                continuation_qualified = (
                    continuation_allowed
                    and len(continuation_values)
                    >= cls.ENLARGEMENT_CONTINUATION_MIN_TAIL_STEPS
                    and continuation_tail_support
                    >= cls.ENLARGEMENT_CONTINUATION_MIN_SCORE
                    and continuation_path_support
                    >= cls.ENLARGEMENT_CONTINUATION_MIN_PATH_SUPPORT
                    and continuation_tail_structure_support
                    >= cls.ENLARGEMENT_CONTINUATION_MIN_SAMPLE_SUPPORT
                )
                if coherent_boundary is not None:
                    growth[side] = coherent_boundary[0]
                    boundary_score = cls._clamp01(
                        0.60 * selected_profile["edge_coherence_score"]
                        + 0.25 * selected_profile["outer_background_contrast"]
                        + 0.15 * selected_profile["background_stability"]
                    )
                    boundary_debug[side] = {
                        "found": True,
                        "growth": growth[side],
                        "score": boundary_score,
                        "confidence": selected_profile["edge_span_support"],
                        "reason": "coherent_outer_edge",
                        "obs_hit": False,
                        "state": "boundary",
                        "continuation_score": continuation_score,
                        "continuation_confidence": 0.0,
                        "continuation_path_support": continuation_path_support,
                        "continuation_tail_support": continuation_tail_support,
                        "continuation_sample_support": continuation_tail_structure_support,
                        "censored": False,
                        "sample_count": sample_count,
                        "boundary_sample_count": len(positions),
                        "continuation_sample_count": 0,
                        "unresolved_sample_count": sample_count - len(positions),
                        "edge_coherence_score": selected_profile["edge_coherence_score"],
                        "edge_span_support": selected_profile["edge_span_support"],
                        "edge_mean_strength": selected_profile["edge_mean_strength"],
                        "edge_orientation_score": selected_profile["edge_orientation_score"],
                        "edge_distance": growth[side],
                        "edge_found": True,
                        "structure_persistence": selected_profile["structure_persistence"],
                        "background_takeover_score": cls._clamp01(
                            selected_profile["background_stability"]
                            * (1.0 - selected_profile["structure_persistence"])
                        ),
                    }
                elif len(positions) >= required:
                    ordered_positions = sorted(position for position, _ in positions)
                    growth[side] = ordered_positions[len(ordered_positions) // 2]
                    boundary_debug[side] = {
                        "found": True,
                        "growth": growth[side],
                        "score": sum(scores) / len(scores),
                        "confidence": len(positions) / sample_count,
                        "reason": "legacy_stable_transition",
                        "obs_hit": False,
                        "state": "boundary",
                        "continuation_score": 0.0,
                        "continuation_confidence": 0.0,
                        "continuation_path_support": 0.0,
                        "continuation_tail_support": 0.0,
                        "continuation_sample_support": 0.0,
                        "censored": False,
                        "sample_count": sample_count,
                        "boundary_sample_count": len(positions),
                        "continuation_sample_count": 0,
                        "unresolved_sample_count": sample_count - len(positions),
                        "edge_coherence_score": selected_profile["edge_coherence_score"],
                        "edge_span_support": selected_profile["edge_span_support"],
                        "edge_mean_strength": selected_profile["edge_mean_strength"],
                        "edge_orientation_score": selected_profile["edge_orientation_score"],
                        "edge_distance": None,
                        "edge_found": False,
                        "structure_persistence": selected_profile["structure_persistence"],
                        "background_takeover_score": cls._clamp01(
                            selected_profile["background_stability"]
                            * (1.0 - selected_profile["structure_persistence"])
                        ),
                    }
                elif continuation_qualified:
                    growth[side] = maximum
                    continuation_confidence = (
                        continuation_tail_support * continuation_path_support
                    )
                    boundary_debug[side] = {
                        "found": False,
                        "growth": maximum,
                        "score": max(scores, default=0.0),
                        "confidence": continuation_confidence,
                        "reason": "structure_continues_to_observation_limit",
                        "obs_hit": True,
                        "state": "continuation_to_limit",
                        "continuation_score": continuation_score,
                        "continuation_confidence": continuation_confidence,
                        "continuation_path_support": continuation_path_support,
                        "continuation_tail_support": continuation_tail_support,
                        "continuation_sample_support": continuation_tail_structure_support,
                        "censored": True,
                        "sample_count": sample_count,
                        "boundary_sample_count": len(positions),
                        "continuation_sample_count": sum(
                            value >= cls.ENLARGEMENT_CONTINUATION_MIN_SCORE
                            for value in continuation_values
                        ),
                        "unresolved_sample_count": sum(
                            value < cls.ENLARGEMENT_CONTINUATION_MIN_SCORE
                            for value in continuation_values
                        ),
                        "edge_coherence_score": selected_profile["edge_coherence_score"],
                        "edge_span_support": selected_profile["edge_span_support"],
                        "edge_mean_strength": selected_profile["edge_mean_strength"],
                        "edge_orientation_score": selected_profile["edge_orientation_score"],
                        "edge_distance": None,
                        "edge_found": False,
                        "structure_persistence": selected_profile["structure_persistence"],
                        "background_takeover_score": cls._clamp01(
                            selected_profile["background_stability"]
                            * (1.0 - selected_profile["structure_persistence"])
                        ),
                    }
                else:
                    growth[side] = maximum
                    boundary_debug[side] = {
                        "found": False,
                        "growth": maximum,
                        "score": max(scores, default=0.0),
                        "confidence": len(positions) / sample_count,
                        "reason": "no_coherent_edge_or_persistent_structure",
                        "obs_hit": True,
                        "state": "unresolved",
                        "continuation_score": 0.0,
                        "continuation_confidence": 0.0,
                        "continuation_path_support": 0.0,
                        "continuation_tail_support": 0.0,
                        "continuation_sample_support": continuation_tail_structure_support,
                        "censored": False,
                        "sample_count": sample_count,
                        "boundary_sample_count": len(positions),
                        "continuation_sample_count": sum(
                            value >= cls.ENLARGEMENT_CONTINUATION_MIN_SCORE
                            for value in continuation_values
                        ),
                        "unresolved_sample_count": sum(
                            value < cls.ENLARGEMENT_CONTINUATION_MIN_SCORE
                            for value in continuation_values
                        ),
                        "edge_coherence_score": selected_profile["edge_coherence_score"],
                        "edge_span_support": selected_profile["edge_span_support"],
                        "edge_mean_strength": selected_profile["edge_mean_strength"],
                        "edge_orientation_score": selected_profile["edge_orientation_score"],
                        "edge_distance": None,
                        "edge_found": False,
                        "structure_persistence": selected_profile["structure_persistence"],
                        "background_takeover_score": cls._clamp01(
                            selected_profile["background_stability"]
                            * (1.0 - selected_profile["structure_persistence"])
                        ),
                    }

            selected_edge_coordinates: dict[str, float] = {
                "left": left - growth["left"],
                "right": right + growth["right"],
                "top": top - growth["top"],
                "bottom": bottom + growth["bottom"],
            }
            device_edges = {
                "left": item.get("prepared_device_left"),
                "right": item.get("prepared_device_right"),
                "top": item.get("prepared_device_top"),
                "bottom": item.get("prepared_device_bottom"),
            }
            device_confidences = {
                "left": item.get("source_device_left_confidence"),
                "right": item.get("source_device_right_confidence"),
                "top": item.get("source_device_top_confidence"),
                "bottom": item.get("source_device_bottom_confidence"),
            }
            device_valid = {
                "left": bool(item.get("source_device_left_valid")),
                "right": bool(item.get("source_device_right_valid")),
                "top": bool(item.get("source_device_top_valid")),
                "bottom": bool(item.get("source_device_bottom_valid")),
            }
            device_distances: dict[str, float | None] = {}
            for side, entry in boundary_debug.items():
                device_edge = device_edges[side]
                candidate_dimension = width if side in ("left", "right") else height
                tolerance = max(
                    cls.ENLARGEMENT_DEVICE_BOUNDARY_TOLERANCE_PX,
                    cls.ENLARGEMENT_DEVICE_BOUNDARY_TOLERANCE_RATIO
                    * candidate_dimension,
                )
                distance_to_device = (
                    abs(selected_edge_coordinates[side] - float(device_edge))
                    if device_valid[side]
                    and isinstance(device_edge, (int, float))
                    else None
                )
                device_distances[side] = distance_to_device
                contaminated = bool(
                    entry["state"] in ("boundary", "continuation_to_limit")
                    and distance_to_device is not None
                    and isinstance(device_confidences[side], (int, float))
                    and float(device_confidences[side])
                    >= cls.DEVICE_BOUNDARY_MIN_SCORE
                    and distance_to_device <= tolerance
                )
                entry["selected_edge_coordinate"] = selected_edge_coordinates[side]
                entry["distance_to_device_boundary"] = distance_to_device
                entry["device_boundary_contaminated"] = contaminated
                entry["boundary_source"] = (
                    "coherent_outer_edge"
                    if entry["reason"] == "coherent_outer_edge"
                    else "legacy_stable_transition"
                    if entry["reason"] == "legacy_stable_transition"
                    else "continuation_to_limit"
                    if entry["state"] == "continuation_to_limit"
                    else "unresolved"
                )
                if contaminated:
                    entry.update({
                        "found": False,
                        "state": "unresolved",
                        "reason": "device_boundary_contamination",
                        "confidence": 0.0,
                        "continuation_confidence": 0.0,
                        "censored": False,
                        "edge_found": False,
                        "boundary_source": "device_boundary_contaminated",
                    })

            raw_growth = dict(growth)
            raw_extent = clipped_box((
                left - raw_growth["left"],
                top - raw_growth["top"],
                right + raw_growth["right"],
                bottom + raw_growth["bottom"],
            )) or semantic
            tolerance = cls.ENLARGEMENT_OBS_BOUNDARY_TOLERANCE_PX
            obs_hits = {
                "left": abs(raw_extent[0] - observation[0]) <= tolerance,
                "right": abs(raw_extent[2] - observation[2]) <= tolerance,
                "top": abs(raw_extent[1] - observation[1]) <= tolerance,
                "bottom": abs(raw_extent[3] - observation[3]) <= tolerance,
            }
            for side, hit in obs_hits.items():
                boundary_debug[side]["obs_hit"] = hit
            horizontal_truncated = (
                boundary_debug["left"]["state"] == "unresolved"
                and boundary_debug["right"]["state"] == "unresolved"
                and obs_hits["left"]
                and obs_hits["right"]
            )
            vertical_truncated = (
                boundary_debug["top"]["state"] == "unresolved"
                and boundary_debug["bottom"]["state"] == "unresolved"
                and obs_hits["top"]
                and obs_hits["bottom"]
            )
            extent_truncated = horizontal_truncated or vertical_truncated

            available_space = {
                "left": max(0.0, left - observation[0]),
                "right": max(0.0, observation[2] - right),
                "top": max(0.0, top - observation[1]),
                "bottom": max(0.0, observation[3] - bottom),
            }
            reconstructed_growth = {side: 0.0 for side in growth}
            reconstructed_confidence = {side: 0.0 for side in growth}
            reconstructed_source = {side: "unresolved" for side in growth}
            completion_clipped = {side: False for side in growth}

            def reconstruct_axis(
                first: str,
                second: str,
            ) -> tuple[bool, float, str]:
                if (
                    boundary_debug[first].get("device_boundary_contaminated")
                    or boundary_debug[second].get("device_boundary_contaminated")
                ):
                    return False, 0.0, "unresolved"
                first_state = str(boundary_debug[first]["state"])
                second_state = str(boundary_debug[second]["state"])
                if first_state == "boundary" and second_state == "boundary":
                    reconstructed_growth[first] = raw_growth[first]
                    reconstructed_growth[second] = raw_growth[second]
                    reconstructed_confidence[first] = boundary_debug[first]["confidence"]
                    reconstructed_confidence[second] = boundary_debug[second]["confidence"]
                    reconstructed_source[first] = "measured"
                    reconstructed_source[second] = "measured"
                    return (
                        True,
                        (reconstructed_confidence[first] + reconstructed_confidence[second]) / 2.0,
                        "measured",
                    )
                if {first_state, second_state} == {"boundary", "unresolved"}:
                    measured, missing = (
                        (first, second)
                        if first_state == "boundary" else (second, first)
                    )
                    requested = float(raw_growth[measured])
                    completed = min(requested, available_space[missing])
                    retained = completed / max(requested, 1e-6)
                    completion_clipped[missing] = completed + 1e-6 < requested
                    reconstructed_growth[measured] = requested
                    reconstructed_growth[missing] = completed
                    reconstructed_confidence[measured] = boundary_debug[measured]["confidence"]
                    reconstructed_confidence[missing] = (
                        boundary_debug[measured]["confidence"]
                        * cls.ENLARGEMENT_MIRRORED_CONFIDENCE_FACTOR
                    )
                    reconstructed_source[measured] = "measured"
                    reconstructed_source[missing] = f"mirrored_from_{measured}"
                    return (
                        retained >= cls.ENLARGEMENT_COMPLETION_MIN_RETAINED_RATIO,
                        (reconstructed_confidence[first] + reconstructed_confidence[second]) / 2.0,
                        "reconstructed",
                    )
                if {first_state, second_state} == {"boundary", "continuation_to_limit"}:
                    for side, state in ((first, first_state), (second, second_state)):
                        reconstructed_growth[side] = raw_growth[side]
                        reconstructed_source[side] = (
                            "measured"
                            if state == "boundary"
                            else "continuation_to_limit"
                        )
                        reconstructed_confidence[side] = (
                            boundary_debug[side]["confidence"]
                            if state == "boundary"
                            else boundary_debug[side]["continuation_confidence"]
                            * cls.ENLARGEMENT_CENSORED_CONFIDENCE_FACTOR
                        )
                    return (
                        True,
                        (reconstructed_confidence[first] + reconstructed_confidence[second]) / 2.0,
                        "partially_censored",
                    )
                if first_state == "continuation_to_limit" and second_state == "continuation_to_limit":
                    for side in (first, second):
                        reconstructed_growth[side] = raw_growth[side]
                        reconstructed_source[side] = "continuation_to_limit"
                        reconstructed_confidence[side] = (
                            boundary_debug[side]["continuation_confidence"]
                            * cls.ENLARGEMENT_CENSORED_CONFIDENCE_FACTOR
                        )
                    return (
                        True,
                        (reconstructed_confidence[first] + reconstructed_confidence[second]) / 2.0,
                        "fully_censored",
                    )
                if {first_state, second_state} == {"continuation_to_limit", "unresolved"}:
                    continuation = (
                        first
                        if first_state == "continuation_to_limit" else second
                    )
                    reconstructed_growth[continuation] = raw_growth[continuation]
                    reconstructed_source[continuation] = "continuation_to_limit"
                    reconstructed_confidence[continuation] = (
                        boundary_debug[continuation]["continuation_confidence"]
                        * cls.ENLARGEMENT_CENSORED_CONFIDENCE_FACTOR
                    )
                    return False, 0.0, "unresolved"
                return False, 0.0, "unresolved"

            horizontal_count = int(boundary_debug["left"]["found"]) + int(boundary_debug["right"]["found"])
            vertical_count = int(boundary_debug["top"]["found"]) + int(boundary_debug["bottom"]["found"])
            horizontal_reliable, horizontal_confidence, horizontal_state = reconstruct_axis(
                "left", "right"
            )
            vertical_reliable, vertical_confidence, vertical_state = reconstruct_axis(
                "top", "bottom"
            )
            extent_reliable = horizontal_reliable and vertical_reliable
            reliability = (
                math.sqrt(max(0.0, horizontal_confidence * vertical_confidence))
                if extent_reliable else 0.0
            )
            used_mirror = any(source.startswith("mirrored_") for source in reconstructed_source.values())
            has_censored_measurement = any(
                entry["state"] == "continuation_to_limit"
                for entry in boundary_debug.values()
            )
            horizontal_censored = any(
                boundary_debug[side]["state"] == "continuation_to_limit"
                for side in ("left", "right")
            )
            vertical_censored = any(
                boundary_debug[side]["state"] == "continuation_to_limit"
                for side in ("top", "bottom")
            )
            fully_measured = (
                horizontal_state == "measured"
                and vertical_state == "measured"
                and not used_mirror
                and not has_censored_measurement
            )
            extent = clipped_box((
                left - reconstructed_growth["left"],
                top - reconstructed_growth["top"],
                right + reconstructed_growth["right"],
                bottom + reconstructed_growth["bottom"],
            )) or semantic
            horizontal_balance = min(reconstructed_growth["left"], reconstructed_growth["right"]) / max(reconstructed_growth["left"], reconstructed_growth["right"], 1.0)
            vertical_balance = min(reconstructed_growth["top"], reconstructed_growth["bottom"]) / max(reconstructed_growth["top"], reconstructed_growth["bottom"], 1.0)
            item.update({
                "visual_extent_bbox": [round(value, 2) for value in extent],
                "visual_extent_width": max(0.0, extent[2] - extent[0]),
                "visual_extent_height": max(0.0, extent[3] - extent[1]),
                "visual_extent_area": max(0.0, extent[2] - extent[0]) * max(0.0, extent[3] - extent[1]),
                "extent_valid": bool(
                    extent_reliable
                    and extent[2] > extent[0]
                    and extent[3] > extent[1]
                ),
                "visual_extent_left_growth_raw": raw_growth["left"],
                "visual_extent_right_growth_raw": raw_growth["right"],
                "visual_extent_top_growth_raw": raw_growth["top"],
                "visual_extent_bottom_growth_raw": raw_growth["bottom"],
                "visual_extent_left_growth_reconstructed": reconstructed_growth["left"],
                "visual_extent_right_growth_reconstructed": reconstructed_growth["right"],
                "visual_extent_top_growth_reconstructed": reconstructed_growth["top"],
                "visual_extent_bottom_growth_reconstructed": reconstructed_growth["bottom"],
                "visual_extent_left_growth": reconstructed_growth["left"],
                "visual_extent_right_growth": reconstructed_growth["right"],
                "visual_extent_top_growth": reconstructed_growth["top"],
                "visual_extent_bottom_growth": reconstructed_growth["bottom"],
                "visual_extent_left_growth_ratio": reconstructed_growth["left"] / max(width, 1e-6),
                "visual_extent_right_growth_ratio": reconstructed_growth["right"] / max(width, 1e-6),
                "visual_extent_top_growth_ratio": reconstructed_growth["top"] / max(height, 1e-6),
                "visual_extent_bottom_growth_ratio": reconstructed_growth["bottom"] / max(height, 1e-6),
                "extent_horizontal_balance": horizontal_balance,
                "extent_vertical_balance": vertical_balance,
                "extent_symmetry": math.sqrt(max(0.0, horizontal_balance * vertical_balance)),
                "extent_boundary_method": "edge_coherent_boundary_or_continuation_v5_4_1",
                "extent_boundary_left_found": boundary_debug["left"]["found"],
                "extent_boundary_right_found": boundary_debug["right"]["found"],
                "extent_boundary_top_found": boundary_debug["top"]["found"],
                "extent_boundary_bottom_found": boundary_debug["bottom"]["found"],
                "extent_boundary_left_score": boundary_debug["left"]["score"],
                "extent_boundary_right_score": boundary_debug["right"]["score"],
                "extent_boundary_top_score": boundary_debug["top"]["score"],
                "extent_boundary_bottom_score": boundary_debug["bottom"]["score"],
                "extent_boundary_left_confidence": boundary_debug["left"]["confidence"],
                "extent_boundary_right_confidence": boundary_debug["right"]["confidence"],
                "extent_boundary_top_confidence": boundary_debug["top"]["confidence"],
                "extent_boundary_bottom_confidence": boundary_debug["bottom"]["confidence"],
                "extent_boundary_left_reason": boundary_debug["left"]["reason"],
                "extent_boundary_right_reason": boundary_debug["right"]["reason"],
                "extent_boundary_top_reason": boundary_debug["top"]["reason"],
                "extent_boundary_bottom_reason": boundary_debug["bottom"]["reason"],
                "extent_left_state": boundary_debug["left"]["state"],
                "extent_right_state": boundary_debug["right"]["state"],
                "extent_top_state": boundary_debug["top"]["state"],
                "extent_bottom_state": boundary_debug["bottom"]["state"],
                "extent_left_continuation_score": boundary_debug["left"]["continuation_score"],
                "extent_right_continuation_score": boundary_debug["right"]["continuation_score"],
                "extent_top_continuation_score": boundary_debug["top"]["continuation_score"],
                "extent_bottom_continuation_score": boundary_debug["bottom"]["continuation_score"],
                "extent_left_continuation_confidence": boundary_debug["left"]["continuation_confidence"],
                "extent_right_continuation_confidence": boundary_debug["right"]["continuation_confidence"],
                "extent_top_continuation_confidence": boundary_debug["top"]["continuation_confidence"],
                "extent_bottom_continuation_confidence": boundary_debug["bottom"]["continuation_confidence"],
                "extent_left_edge_coherence_score": boundary_debug["left"]["edge_coherence_score"],
                "extent_right_edge_coherence_score": boundary_debug["right"]["edge_coherence_score"],
                "extent_top_edge_coherence_score": boundary_debug["top"]["edge_coherence_score"],
                "extent_bottom_edge_coherence_score": boundary_debug["bottom"]["edge_coherence_score"],
                "extent_left_edge_span_support": boundary_debug["left"]["edge_span_support"],
                "extent_right_edge_span_support": boundary_debug["right"]["edge_span_support"],
                "extent_top_edge_span_support": boundary_debug["top"]["edge_span_support"],
                "extent_bottom_edge_span_support": boundary_debug["bottom"]["edge_span_support"],
                "extent_left_edge_mean_strength": boundary_debug["left"]["edge_mean_strength"],
                "extent_right_edge_mean_strength": boundary_debug["right"]["edge_mean_strength"],
                "extent_top_edge_mean_strength": boundary_debug["top"]["edge_mean_strength"],
                "extent_bottom_edge_mean_strength": boundary_debug["bottom"]["edge_mean_strength"],
                "extent_left_edge_distance": boundary_debug["left"]["edge_distance"],
                "extent_right_edge_distance": boundary_debug["right"]["edge_distance"],
                "extent_top_edge_distance": boundary_debug["top"]["edge_distance"],
                "extent_bottom_edge_distance": boundary_debug["bottom"]["edge_distance"],
                "extent_left_edge_found": boundary_debug["left"]["edge_found"],
                "extent_right_edge_found": boundary_debug["right"]["edge_found"],
                "extent_top_edge_found": boundary_debug["top"]["edge_found"],
                "extent_bottom_edge_found": boundary_debug["bottom"]["edge_found"],
                "extent_left_edge_orientation_score": boundary_debug["left"]["edge_orientation_score"],
                "extent_right_edge_orientation_score": boundary_debug["right"]["edge_orientation_score"],
                "extent_top_edge_orientation_score": boundary_debug["top"]["edge_orientation_score"],
                "extent_bottom_edge_orientation_score": boundary_debug["bottom"]["edge_orientation_score"],
                "extent_left_boundary_source": boundary_debug["left"]["boundary_source"],
                "extent_right_boundary_source": boundary_debug["right"]["boundary_source"],
                "extent_top_boundary_source": boundary_debug["top"]["boundary_source"],
                "extent_bottom_boundary_source": boundary_debug["bottom"]["boundary_source"],
                "extent_left_selected_edge_coordinate": selected_edge_coordinates["left"],
                "extent_right_selected_edge_coordinate": selected_edge_coordinates["right"],
                "extent_top_selected_edge_coordinate": selected_edge_coordinates["top"],
                "extent_bottom_selected_edge_coordinate": selected_edge_coordinates["bottom"],
                "extent_left_distance_to_device_boundary": device_distances["left"],
                "extent_right_distance_to_device_boundary": device_distances["right"],
                "extent_top_distance_to_device_boundary": device_distances["top"],
                "extent_bottom_distance_to_device_boundary": device_distances["bottom"],
                "extent_left_device_boundary_contaminated": boundary_debug["left"]["device_boundary_contaminated"],
                "extent_right_device_boundary_contaminated": boundary_debug["right"]["device_boundary_contaminated"],
                "extent_top_device_boundary_contaminated": boundary_debug["top"]["device_boundary_contaminated"],
                "extent_bottom_device_boundary_contaminated": boundary_debug["bottom"]["device_boundary_contaminated"],
                "extent_device_boundary_contaminated_side_count": sum(
                    entry["device_boundary_contaminated"]
                    for entry in boundary_debug.values()
                ),
                "extent_edge_coherent_side_count": sum(
                    entry["edge_found"] for entry in boundary_debug.values()
                ),
                "extent_mean_edge_coherence": sum(
                    entry["edge_coherence_score"]
                    for entry in boundary_debug.values()
                ) / len(boundary_debug),
                "extent_left_censored": boundary_debug["left"]["censored"],
                "extent_right_censored": boundary_debug["right"]["censored"],
                "extent_top_censored": boundary_debug["top"]["censored"],
                "extent_bottom_censored": boundary_debug["bottom"]["censored"],
                "extent_censored_side_count": sum(
                    entry["censored"] for entry in boundary_debug.values()
                ),
                "extent_obs_hit_left": obs_hits["left"],
                "extent_obs_hit_right": obs_hits["right"],
                "extent_obs_hit_top": obs_hits["top"],
                "extent_obs_hit_bottom": obs_hits["bottom"],
                "extent_obs_boundary_hit_count": sum(obs_hits.values()),
                "extent_horizontal_boundary_count": horizontal_count,
                "extent_vertical_boundary_count": vertical_count,
                "extent_horizontal_reliable": horizontal_reliable,
                "extent_vertical_reliable": vertical_reliable,
                "extent_horizontal_censored": horizontal_censored,
                "extent_vertical_censored": vertical_censored,
                "extent_horizontal_state": horizontal_state,
                "extent_vertical_state": vertical_state,
                "extent_left_source": reconstructed_source["left"],
                "extent_right_source": reconstructed_source["right"],
                "extent_top_source": reconstructed_source["top"],
                "extent_bottom_source": reconstructed_source["bottom"],
                "extent_completion_clipped_left": completion_clipped["left"],
                "extent_completion_clipped_right": completion_clipped["right"],
                "extent_completion_clipped_top": completion_clipped["top"],
                "extent_completion_clipped_bottom": completion_clipped["bottom"],
                "extent_completion_clip_count": sum(completion_clipped.values()),
                "extent_reconstructed_left_confidence": reconstructed_confidence["left"],
                "extent_reconstructed_right_confidence": reconstructed_confidence["right"],
                "extent_reconstructed_top_confidence": reconstructed_confidence["top"],
                "extent_reconstructed_bottom_confidence": reconstructed_confidence["bottom"],
                "extent_horizontal_confidence": horizontal_confidence,
                "extent_vertical_confidence": vertical_confidence,
                "extent_horizontal_truncated": horizontal_truncated,
                "extent_vertical_truncated": vertical_truncated,
                "extent_truncated": extent_truncated,
                "extent_observation_censored": has_censored_measurement,
                "extent_has_censored_measurement": has_censored_measurement,
                "extent_fully_measured": fully_measured,
                "extent_width_is_lower_bound": horizontal_censored,
                "extent_height_is_lower_bound": vertical_censored,
                "extent_area_is_lower_bound": horizontal_censored or vertical_censored,
                "extent_boundary_reliability": reliability,
                "extent_reliable": extent_reliable,
                "enlargement_extent_reason": (
                    "device_boundary_contamination" if any(
                        entry["device_boundary_contaminated"]
                        for entry in boundary_debug.values()
                    )
                    else "observation_boundary_truncated" if extent_truncated
                    else "censored_continuation_extent" if extent_reliable and has_censored_measurement
                    else "partial_boundary_reconstructed" if extent_reliable and used_mirror
                    else "stable_boundary_extent" if extent_reliable
                    else "insufficient_axis_boundary"
                ),
                "extent_side_support": {
                    side: entry["confidence"]
                    for side, entry in boundary_debug.items()
                },
                "extent_boundary_debug": boundary_debug,
            })

        for item in evidence:
            detect_visual_footprint(item)

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

        def build_enlargement_card_observation(
            item: dict[str, Any],
            siblings: list[dict[str, Any]],
        ) -> tuple[tuple[float, float, float, float] | None, bool]:
            semantic = clipped_box(item.get("prepared_bbox"))
            visual_cell = clipped_box(item.get("visual_cell_bbox"))
            if semantic is None or visual_cell is None:
                item.update({
                    "enlargement_card_layout": "sibling_protected_core_v5_3_2",
                    "enlargement_card_observation_valid": False,
                    "enlargement_extent_observation_source": "pure_card_v5_3",
                })
                return None, False
            left, top, right, bottom = semantic
            width, height = right - left, bottom - top
            legacy_observation = clipped_box((
                max(visual_cell[0], left - width * cls.ENLARGEMENT_LOCAL_MARGIN_X),
                max(visual_cell[1], top - height * cls.ENLARGEMENT_LOCAL_MARGIN_Y),
                min(visual_cell[2], right + width * cls.ENLARGEMENT_LOCAL_MARGIN_X),
                min(visual_cell[3], bottom + height * cls.ENLARGEMENT_LOCAL_MARGIN_Y),
            ))
            item["enlargement_local_observation_bbox"] = [
                round(value, 2) for value in legacy_observation
            ] if legacy_observation else None
            center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
            centers: list[tuple[int, float, float]] = []
            sibling_boxes: dict[int, tuple[float, float, float, float]] = {}
            for sibling in siblings:
                if sibling is item:
                    continue
                sibling_box = clipped_box(sibling.get("prepared_bbox"))
                if sibling_box is not None:
                    sibling_boxes[int(sibling["index"])] = sibling_box
                    centers.append((
                        int(sibling["index"]),
                        (sibling_box[0] + sibling_box[2]) / 2.0,
                        (sibling_box[1] + sibling_box[3]) / 2.0,
                    ))

            def sibling_protected_core(
                sibling_box: tuple[float, float, float, float],
            ) -> tuple[float, float, float, float]:
                sibling_left, sibling_top, sibling_right, sibling_bottom = sibling_box
                sibling_width = sibling_right - sibling_left
                sibling_height = sibling_bottom - sibling_top
                horizontal_inset = (
                    0.5
                    * (1.0 - cls.ENLARGEMENT_SIBLING_PROTECTED_CORE_X)
                    * sibling_width
                )
                vertical_inset = (
                    0.5
                    * (1.0 - cls.ENLARGEMENT_SIBLING_PROTECTED_CORE_Y)
                    * sibling_height
                )
                return (
                    sibling_left + horizontal_inset,
                    sibling_top + vertical_inset,
                    sibling_right - horizontal_inset,
                    sibling_bottom - vertical_inset,
                )

            protected_cores = {
                index: sibling_protected_core(sibling_box)
                for index, sibling_box in sibling_boxes.items()
            }

            def nearest(predicate: Any, distance: Any) -> tuple[int | None, float | None]:
                values = [(index, distance(x, y)) for index, x, y in centers if predicate(x, y)]
                return min(values, key=lambda value: value[1]) if values else (None, None)

            left_index, left_distance = nearest(lambda x, y: x < center_x, lambda x, y: center_x - x)
            right_index, right_distance = nearest(lambda x, y: x > center_x, lambda x, y: x - center_x)
            top_index, top_distance = nearest(lambda x, y: y < center_y, lambda x, y: center_y - y)
            bottom_index, bottom_distance = nearest(lambda x, y: y > center_y, lambda x, y: y - center_y)
            left_core = protected_cores.get(left_index) if left_index is not None else None
            right_core = protected_cores.get(right_index) if right_index is not None else None
            top_core = protected_cores.get(top_index) if top_index is not None else None
            bottom_core = protected_cores.get(bottom_index) if bottom_index is not None else None
            desired_cell_left = left - width * cls.ENLARGEMENT_CARD_CELL_OUTER_X
            desired_cell_right = right + width * cls.ENLARGEMENT_CARD_CELL_OUTER_X
            desired_cell_top = top - height * cls.ENLARGEMENT_CARD_CELL_OUTER_Y
            desired_cell_bottom = bottom + height * cls.ENLARGEMENT_CARD_CELL_OUTER_Y
            cell_left = (
                min(left, max(desired_cell_left, left_core[2]))
                if left_core is not None else desired_cell_left
            )
            cell_right = (
                max(right, min(desired_cell_right, right_core[0]))
                if right_core is not None else desired_cell_right
            )
            cell_top = (
                min(top, max(desired_cell_top, top_core[3]))
                if top_core is not None else desired_cell_top
            )
            cell_bottom = (
                max(bottom, min(desired_cell_bottom, bottom_core[1]))
                if bottom_core is not None else desired_cell_bottom
            )
            card_cell = clipped_box((
                max(visual_cell[0], cell_left),
                max(visual_cell[1], cell_top),
                min(visual_cell[2], cell_right),
                min(visual_cell[3], cell_bottom),
            ))
            desired_observation = (
                left - width * cls.ENLARGEMENT_CARD_OBS_MARGIN_X,
                top - height * cls.ENLARGEMENT_CARD_OBS_MARGIN_Y,
                right + width * cls.ENLARGEMENT_CARD_OBS_MARGIN_X,
                bottom + height * cls.ENLARGEMENT_CARD_OBS_MARGIN_Y,
            )
            observation = clipped_box((
                max(desired_observation[0], card_cell[0]) if card_cell else left,
                max(desired_observation[1], card_cell[1]) if card_cell else top,
                min(desired_observation[2], card_cell[2]) if card_cell else right,
                min(desired_observation[3], card_cell[3]) if card_cell else bottom,
            ))
            contains_semantic = bool(
                card_cell is not None
                and card_cell[0] <= left and card_cell[1] <= top
                and card_cell[2] >= right and card_cell[3] >= bottom
            )
            contains_observation = bool(
                observation is not None
                and observation[0] <= left and observation[1] <= top
                and observation[2] >= right and observation[3] >= bottom
            )
            contains_other_center = bool(
                observation is not None
                and any(
                    observation[0] <= sibling_x <= observation[2]
                    and observation[1] <= sibling_y <= observation[3]
                    for _, sibling_x, sibling_y in centers
                )
            )
            def positive_intersection(
                first: tuple[float, float, float, float],
                second: tuple[float, float, float, float],
            ) -> bool:
                return (
                    min(first[2], second[2]) > max(first[0], second[0])
                    and min(first[3], second[3]) > max(first[1], second[1])
                )

            intersects_other_sibling_core = bool(
                observation is not None
                and any(
                    positive_intersection(observation, protected_core)
                    for protected_core in protected_cores.values()
                )
            )
            left_semantic_gap = max(0.0, left - sibling_boxes[left_index][2]) if left_index is not None else None
            right_semantic_gap = max(0.0, sibling_boxes[right_index][0] - right) if right_index is not None else None
            top_semantic_gap = max(0.0, top - sibling_boxes[top_index][3]) if top_index is not None else None
            bottom_semantic_gap = max(0.0, sibling_boxes[bottom_index][1] - bottom) if bottom_index is not None else None
            left_protected_gap = left - left_core[2] if left_core is not None else None
            right_protected_gap = right_core[0] - right if right_core is not None else None
            top_protected_gap = top - top_core[3] if top_core is not None else None
            bottom_protected_gap = bottom_core[1] - bottom if bottom_core is not None else None
            valid = (
                contains_semantic
                and contains_observation
                and not contains_other_center
                and not intersects_other_sibling_core
            )
            item.update({
                "enlargement_card_layout": "sibling_protected_core_v5_3_2",
                "enlargement_card_cell_bbox": [round(value, 2) for value in card_cell] if card_cell else None,
                "enlargement_card_observation_bbox": [round(value, 2) for value in observation] if observation else None,
                "enlargement_card_left_neighbor_index": left_index,
                "enlargement_card_right_neighbor_index": right_index,
                "enlargement_card_top_neighbor_index": top_index,
                "enlargement_card_bottom_neighbor_index": bottom_index,
                "enlargement_card_left_neighbor_protected_core_bbox": [round(value, 2) for value in left_core] if left_core else None,
                "enlargement_card_right_neighbor_protected_core_bbox": [round(value, 2) for value in right_core] if right_core else None,
                "enlargement_card_top_neighbor_protected_core_bbox": [round(value, 2) for value in top_core] if top_core else None,
                "enlargement_card_bottom_neighbor_protected_core_bbox": [round(value, 2) for value in bottom_core] if bottom_core else None,
                "enlargement_card_left_boundary_source": "sibling_protected_core" if left_core is not None else "semantic_outer",
                "enlargement_card_right_boundary_source": "sibling_protected_core" if right_core is not None else "semantic_outer",
                "enlargement_card_top_boundary_source": "sibling_protected_core" if top_core is not None else "semantic_outer",
                "enlargement_card_bottom_boundary_source": "sibling_protected_core" if bottom_core is not None else "semantic_outer",
                "enlargement_card_left_sibling_gap": left_semantic_gap,
                "enlargement_card_right_sibling_gap": right_semantic_gap,
                "enlargement_card_top_sibling_gap": top_semantic_gap,
                "enlargement_card_bottom_sibling_gap": bottom_semantic_gap,
                "enlargement_card_left_protected_gap": left_protected_gap,
                "enlargement_card_right_protected_gap": right_protected_gap,
                "enlargement_card_top_protected_gap": top_protected_gap,
                "enlargement_card_bottom_protected_gap": bottom_protected_gap,
                "enlargement_card_left_peripheral_allowance": (left_protected_gap - left_semantic_gap) if left_protected_gap is not None and left_semantic_gap is not None else None,
                "enlargement_card_right_peripheral_allowance": (right_protected_gap - right_semantic_gap) if right_protected_gap is not None and right_semantic_gap is not None else None,
                "enlargement_card_top_peripheral_allowance": (top_protected_gap - top_semantic_gap) if top_protected_gap is not None and top_semantic_gap is not None else None,
                "enlargement_card_bottom_peripheral_allowance": (bottom_protected_gap - bottom_semantic_gap) if bottom_protected_gap is not None and bottom_semantic_gap is not None else None,
                "enlargement_card_left_margin": (left - card_cell[0]) if card_cell else 0.0,
                "enlargement_card_right_margin": (card_cell[2] - right) if card_cell else 0.0,
                "enlargement_card_top_margin": (top - card_cell[1]) if card_cell else 0.0,
                "enlargement_card_bottom_margin": (card_cell[3] - bottom) if card_cell else 0.0,
                "enlargement_card_obs_left_margin": (left - observation[0]) if observation else 0.0,
                "enlargement_card_obs_right_margin": (observation[2] - right) if observation else 0.0,
                "enlargement_card_obs_top_margin": (top - observation[1]) if observation else 0.0,
                "enlargement_card_obs_bottom_margin": (observation[3] - bottom) if observation else 0.0,
                "enlargement_card_observation_width": (observation[2] - observation[0]) if observation else 0.0,
                "enlargement_card_observation_height": (observation[3] - observation[1]) if observation else 0.0,
                "enlargement_card_observation_to_semantic_width_ratio": ((observation[2] - observation[0]) / max(width, 1e-6)) if observation else 0.0,
                "enlargement_card_observation_to_semantic_height_ratio": ((observation[3] - observation[1]) / max(height, 1e-6)) if observation else 0.0,
                "enlargement_card_cell_contains_semantic": contains_semantic,
                "enlargement_card_observation_contains_semantic": contains_observation,
                "enlargement_card_observation_contains_other_sibling_center": contains_other_center,
                "enlargement_card_observation_intersects_other_sibling_core": intersects_other_sibling_core,
                "enlargement_card_observation_valid": valid,
                "enlargement_extent_observation_source": "pure_card_v5_3",
            })
            return observation, valid

        def recover_independent_current_container(
            item: dict[str, Any],
            observation: tuple[float, float, float, float] | None,
            siblings: list[dict[str, Any]],
        ) -> None:
            """Measure the nearest enclosing card body without V5.4 extent state."""
            semantic = clipped_box(item.get("prepared_bbox"))

            def unavailable(reason: str) -> None:
                item.update({
                    "recovered_current_container_bbox": None,
                    "recovered_current_container_width": None,
                    "recovered_current_container_height": None,
                    "recovered_current_container_area": None,
                    "recovered_current_container_valid": False,
                    "recovered_current_container_confidence": 0.0,
                    "recovered_current_container_source": "unavailable",
                    "recovered_container_left": None,
                    "recovered_container_right": None,
                    "recovered_container_top": None,
                    "recovered_container_bottom": None,
                    "semantic_coverage_w": None,
                    "semantic_coverage_h": None,
                    "recovered_container_left_padding_ratio": None,
                    "recovered_container_right_padding_ratio": None,
                    "recovered_container_top_padding_ratio": None,
                    "recovered_container_bottom_padding_ratio": None,
                    **{
                        f"natural_container_{side}_state": "unresolved"
                        for side in ("left", "right", "top", "bottom")
                    },
                    **{
                        f"natural_container_{side}_reason": reason
                        for side in ("left", "right", "top", "bottom")
                    },
                    **{
                        f"natural_container_{side}_score": 0.0
                        for side in ("left", "right", "top", "bottom")
                    },
                    **{
                        f"natural_container_{side}_distance": None
                        for side in ("left", "right", "top", "bottom")
                    },
                    **{
                        f"natural_container_{side}_scan_mode": "unresolved"
                        for side in ("left", "right", "top", "bottom")
                    },
                    **{
                        f"natural_container_{side}_selected_scan_step": None
                        for side in ("left", "right", "top", "bottom")
                    },
                    **{
                        f"natural_container_{side}_{metric}": 0.0
                        for side in ("left", "right", "top", "bottom")
                        for metric in (
                            "span_support",
                            "mean_oriented_strength",
                            "inside_outside_contrast",
                        )
                    },
                    "natural_container_dense_boundary_side_count": 0,
                })

            if semantic is None or observation is None:
                unavailable("invalid_geometry")
                return
            search = clipped_box(observation)
            if search is None:
                unavailable("invalid_geometry")
                return
            for boundary in (
                clipped_box(item.get("visual_cell_bbox")),
                clipped_box(item.get("prepared_montage_tile_bbox")),
            ):
                if boundary is None:
                    continue
                search = clipped_box((
                    max(search[0], boundary[0]),
                    max(search[1], boundary[1]),
                    min(search[2], boundary[2]),
                    min(search[3], boundary[3]),
                ))
                if search is None:
                    unavailable("invalid_geometry")
                    return
            left, top, right, bottom = semantic
            if not (
                search[0] <= left and search[1] <= top
                and search[2] >= right and search[3] >= bottom
            ):
                unavailable("invalid_geometry")
                return
            width = max(right - left, 1e-6)
            height = max(bottom - top, 1e-6)
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            sample_count = max(5, min(15, int(round(max(width, height) / 20.0))))
            gradient_radius = cls.ENLARGEMENT_EDGE_COHERENCE_GRADIENT_RADIUS

            def pixel_at(x: int, y: int) -> tuple[int, int, int]:
                return pixel_data[
                    max(int(search[0]), min(int(search[2]) - 1, x)),
                    max(int(search[1]), min(int(search[3]) - 1, y)),
                ]

            def luminance(color: tuple[int, int, int]) -> float:
                return (
                    0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                ) / 255.0

            def oriented_strength(
                first: tuple[int, int, int],
                second: tuple[int, int, int],
            ) -> float:
                luma_gradient = abs(luminance(first) - luminance(second))
                color_gradient = math.sqrt(sum(
                    ((first[channel] - second[channel]) / 255.0) ** 2
                    for channel in range(3)
                ) / 3.0)
                return cls._clamp01(
                    0.45 * cls._clamp01(
                        luma_gradient / cls.ENLARGEMENT_EDGE_COHERENCE_LUMINANCE_NORMALIZER
                    )
                    + 0.55 * cls._clamp01(
                        color_gradient / cls.ENLARGEMENT_EDGE_COHERENCE_COLOR_NORMALIZER
                    )
                )

            def point(side: str, distance: int, fraction: float) -> tuple[int, int]:
                if side == "left":
                    coordinate = left - distance
                    parallel = top + fraction * height
                    return int(round(coordinate)), int(round(parallel))
                if side == "right":
                    coordinate = right + distance
                    parallel = top + fraction * height
                    return int(round(coordinate)), int(round(parallel))
                if side == "top":
                    coordinate = top - distance
                    parallel = left + fraction * width
                    return int(round(parallel)), int(round(coordinate))
                coordinate = bottom + distance
                parallel = left + fraction * width
                return int(round(parallel)), int(round(coordinate))

            def profile(side: str, distance: int) -> dict[str, float]:
                strengths: list[float] = []
                normal_values: list[float] = []
                orientations: list[float] = []
                inset = cls.ENLARGEMENT_EDGE_SAMPLE_INSET
                for sample_index in range(sample_count):
                    fraction = inset + (1.0 - 2.0 * inset) * (sample_index + 0.5) / sample_count
                    x, y = point(side, distance, fraction)
                    if side == "left":
                        inside, outside = pixel_at(x + gradient_radius, y), pixel_at(x - gradient_radius, y)
                        tangent_first, tangent_second = pixel_at(x, y - gradient_radius), pixel_at(x, y + gradient_radius)
                    elif side == "right":
                        inside, outside = pixel_at(x - gradient_radius, y), pixel_at(x + gradient_radius, y)
                        tangent_first, tangent_second = pixel_at(x, y - gradient_radius), pixel_at(x, y + gradient_radius)
                    elif side == "top":
                        inside, outside = pixel_at(x, y + gradient_radius), pixel_at(x, y - gradient_radius)
                        tangent_first, tangent_second = pixel_at(x - gradient_radius, y), pixel_at(x + gradient_radius, y)
                    else:
                        inside, outside = pixel_at(x, y - gradient_radius), pixel_at(x, y + gradient_radius)
                        tangent_first, tangent_second = pixel_at(x - gradient_radius, y), pixel_at(x + gradient_radius, y)
                    normal = oriented_strength(inside, outside)
                    tangent = oriented_strength(tangent_first, tangent_second)
                    orientation = normal / max(normal + tangent, 1e-6)
                    normal_values.append(normal)
                    strengths.append(normal * orientation)
                    orientations.append(orientation)
                span_support = sum(
                    value >= cls.NATURAL_CONTAINER_MIN_ORIENTED_STRENGTH
                    for value in strengths
                ) / max(len(strengths), 1)
                mean_strength = sum(strengths) / max(len(strengths), 1)
                inside_outside_contrast = sum(normal_values) / max(len(normal_values), 1)
                return {
                    "score": cls._clamp01(
                        0.55 * span_support
                        + 0.30 * mean_strength
                        + 0.15 * inside_outside_contrast
                    ),
                    "span_support": span_support,
                    "mean_strength": mean_strength,
                    "orientation": sum(orientations) / max(len(orientations), 1),
                }

            sibling_centers = [
                (
                    (box[0] + box[2]) / 2.0,
                    (box[1] + box[3]) / 2.0,
                )
                for sibling in siblings
                if sibling is not item
                for box in [clipped_box(sibling.get("prepared_bbox"))]
                if box is not None
            ]
            protected_cores = [
                core for field in (
                    "enlargement_card_left_neighbor_protected_core_bbox",
                    "enlargement_card_right_neighbor_protected_core_bbox",
                    "enlargement_card_top_neighbor_protected_core_bbox",
                    "enlargement_card_bottom_neighbor_protected_core_bbox",
                )
                for core in [clipped_box(item.get(field))]
                if core is not None
            ]
            device_edges = {
                "left": item.get("prepared_device_left"),
                "right": item.get("prepared_device_right"),
                "top": item.get("prepared_device_top"),
                "bottom": item.get("prepared_device_bottom"),
            }
            device_valid = {
                "left": bool(item.get("source_device_left_valid")),
                "right": bool(item.get("source_device_right_valid")),
                "top": bool(item.get("source_device_top_valid")),
                "bottom": bool(item.get("source_device_bottom_valid")),
            }
            device_confidence = {
                "left": item.get("source_device_left_confidence"),
                "right": item.get("source_device_right_confidence"),
                "top": item.get("source_device_top_confidence"),
                "bottom": item.get("source_device_bottom_confidence"),
            }
            side_results: dict[str, dict[str, Any]] = {}
            for side in ("left", "right", "top", "bottom"):
                maximum = int(round(
                    left - search[0] if side == "left"
                    else search[2] - right if side == "right"
                    else top - search[1] if side == "top"
                    else search[3] - bottom
                ))
                device_edge = device_edges[side]
                semantic_edge = left if side == "left" else right if side == "right" else top if side == "top" else bottom
                if device_valid[side] and isinstance(device_edge, (int, float)):
                    outward_distance = (
                        semantic_edge - float(device_edge)
                        if side in ("left", "top") else float(device_edge) - semantic_edge
                    )
                    if outward_distance >= 0.0:
                        maximum = min(maximum, int(round(outward_distance)))
                if maximum < 1:
                    side_results[side] = {
                        "state": "unresolved", "reason": "insufficient_scan_space",
                        "score": 0.0, "distance": None, "coordinate": None,
                        "scan_mode": "unresolved", "selected_scan_step": None,
                        "span_support": 0.0, "mean_strength": 0.0,
                        "inside_outside_contrast": 0.0,
                    }
                    continue
                dense_limit = min(
                    maximum,
                    cls.NATURAL_CONTAINER_DENSE_SCAN_RADIUS_PX,
                )
                coarse_step = max(
                    2,
                    min(4, int(round(min(width, height) * 0.025))),
                )
                distances = list(range(1, dense_limit + 1))
                distances.extend(range(
                    cls.NATURAL_CONTAINER_DENSE_SCAN_RADIUS_PX + coarse_step,
                    maximum + 1,
                    coarse_step,
                ))
                distances.append(maximum)
                distances = sorted(set(distances))
                profiles: dict[int, dict[str, float]] = {}

                def at(distance: int) -> dict[str, float]:
                    if distance not in profiles:
                        profiles[distance] = profile(side, distance)
                    return profiles[distance]

                selected: tuple[int, dict[str, float]] | None = None
                for position, distance in enumerate(distances):
                    current = at(distance)
                    adjacent = [
                        at(distances[neighbor])["score"]
                        for neighbor in range(
                            max(0, position - cls.NATURAL_CONTAINER_LOCAL_RIDGE_RADIUS_STEPS),
                            min(len(distances), position + cls.NATURAL_CONTAINER_LOCAL_RIDGE_RADIUS_STEPS + 1),
                        )
                    ]
                    if (
                        current["score"] >= cls.NATURAL_CONTAINER_MIN_EDGE_SCORE
                        and current["span_support"] >= cls.NATURAL_CONTAINER_MIN_SPAN_SUPPORT
                        and current["mean_strength"] >= cls.NATURAL_CONTAINER_MIN_ORIENTED_STRENGTH
                        and current["score"] + cls.NATURAL_CONTAINER_LOCAL_RIDGE_TOLERANCE
                        >= max(adjacent)
                    ):
                        selected = (distance, current)
                        break
                if selected is None:
                    side_results[side] = {
                        "state": "unresolved", "reason": "no_qualifying_container_boundary",
                        "score": 0.0, "distance": None, "coordinate": None,
                        "scan_mode": "unresolved", "selected_scan_step": None,
                        "span_support": 0.0, "mean_strength": 0.0,
                        "inside_outside_contrast": 0.0,
                    }
                    continue
                distance, selected_profile = selected
                coordinate = semantic_edge - distance if side in ("left", "top") else semantic_edge + distance
                core_overlap = any(
                    core[0] < coordinate < core[2]
                    if side in ("left", "right") else core[1] < coordinate < core[3]
                    for core in protected_cores
                )
                beyond_sibling_center = any(
                    center[0] < center_x and coordinate <= center[0]
                    if side == "left"
                    else center[0] > center_x and coordinate >= center[0]
                    if side == "right"
                    else center[1] < center_y and coordinate <= center[1]
                    if side == "top"
                    else center[1] > center_y and coordinate >= center[1]
                    for center in sibling_centers
                )
                candidate_dimension = width if side in ("left", "right") else height
                tolerance = max(
                    cls.ENLARGEMENT_DEVICE_BOUNDARY_TOLERANCE_PX,
                    cls.ENLARGEMENT_DEVICE_BOUNDARY_TOLERANCE_RATIO * candidate_dimension,
                )
                device_contaminated = bool(
                    device_valid[side]
                    and isinstance(device_edge, (int, float))
                    and isinstance(device_confidence[side], (int, float))
                    and float(device_confidence[side]) >= cls.DEVICE_BOUNDARY_MIN_SCORE
                    and abs(coordinate - float(device_edge)) <= tolerance
                )
                if device_contaminated:
                    reason = "device_boundary_contamination"
                elif core_overlap or beyond_sibling_center:
                    reason = "sibling_ownership_rejected"
                else:
                    side_results[side] = {
                        "state": "container_boundary",
                        "reason": "nearest_qualifying_container_boundary",
                        "score": selected_profile["score"],
                        "distance": float(distance),
                        "coordinate": float(coordinate),
                        "scan_mode": (
                            "dense"
                            if distance <= cls.NATURAL_CONTAINER_DENSE_SCAN_RADIUS_PX
                            else "coarse"
                        ),
                        "selected_scan_step": (
                            1
                            if distance <= cls.NATURAL_CONTAINER_DENSE_SCAN_RADIUS_PX
                            else coarse_step
                        ),
                        "span_support": selected_profile["span_support"],
                        "mean_strength": selected_profile["mean_strength"],
                        "inside_outside_contrast": selected_profile["inside_outside_contrast"],
                    }
                    continue
                side_results[side] = {
                    "state": "unresolved", "reason": reason,
                    "score": selected_profile["score"], "distance": None,
                    "coordinate": float(coordinate),
                    "scan_mode": "unresolved", "selected_scan_step": None,
                    "span_support": selected_profile["span_support"],
                    "mean_strength": selected_profile["mean_strength"],
                    "inside_outside_contrast": selected_profile["inside_outside_contrast"],
                }

            valid = all(
                side_results[side]["state"] == "container_boundary"
                for side in ("left", "right", "top", "bottom")
            )
            recovered = (
                side_results["left"]["coordinate"],
                side_results["top"]["coordinate"],
                side_results["right"]["coordinate"],
                side_results["bottom"]["coordinate"],
            ) if valid else None
            recovered_width = recovered[2] - recovered[0] if recovered else None
            recovered_height = recovered[3] - recovered[1] if recovered else None
            confidence = min(
                side_results[side]["score"] for side in ("left", "right", "top", "bottom")
            ) if valid else 0.0
            item.update({
                "recovered_current_container_bbox": [round(value, 2) for value in recovered] if recovered else None,
                "recovered_current_container_width": recovered_width,
                "recovered_current_container_height": recovered_height,
                "recovered_current_container_area": recovered_width * recovered_height if recovered_width is not None and recovered_height is not None else None,
                "recovered_current_container_valid": valid,
                "recovered_current_container_confidence": confidence,
                "recovered_current_container_source": "independent_pixel_boundary_v5_5_1" if valid else "unavailable",
                "recovered_container_left": recovered[0] if recovered else None,
                "recovered_container_right": recovered[2] if recovered else None,
                "recovered_container_top": recovered[1] if recovered else None,
                "recovered_container_bottom": recovered[3] if recovered else None,
                "semantic_coverage_w": width / max(recovered_width, 1e-6) if recovered_width is not None else None,
                "semantic_coverage_h": height / max(recovered_height, 1e-6) if recovered_height is not None else None,
                "recovered_container_left_padding_ratio": (left - recovered[0]) / width if recovered else None,
                "recovered_container_right_padding_ratio": (recovered[2] - right) / width if recovered else None,
                "recovered_container_top_padding_ratio": (top - recovered[1]) / height if recovered else None,
                "recovered_container_bottom_padding_ratio": (recovered[3] - bottom) / height if recovered else None,
                **{
                    f"natural_container_{side}_state": side_results[side]["state"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_reason": side_results[side]["reason"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_score": side_results[side]["score"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_distance": side_results[side]["distance"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_scan_mode": side_results[side]["scan_mode"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_selected_scan_step": side_results[side]["selected_scan_step"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_span_support": side_results[side]["span_support"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_mean_oriented_strength": side_results[side]["mean_strength"]
                    for side in side_results
                },
                **{
                    f"natural_container_{side}_inside_outside_contrast": side_results[side]["inside_outside_contrast"]
                    for side in side_results
                },
                "natural_container_dense_boundary_side_count": sum(
                    side_results[side]["state"] == "container_boundary"
                    and side_results[side]["scan_mode"] == "dense"
                    for side in side_results
                ),
            })

        for sibling_set in sibling_sets:
            members = [by_index[index] for index in sibling_set if index in by_index]
            for item in members:
                card_observation, valid = build_enlargement_card_observation(item, members)
                if valid and card_observation is not None:
                    recover_independent_current_container(item, card_observation, members)
                    measure_direct_extent(item, card_observation)
                else:
                    recover_independent_current_container(item, None, members)
                    item["extent_reliable"] = False
                    item["enlargement_extent_reason"] = "invalid_geometry"

        def independent_container_sample(
            item: dict[str, Any],
        ) -> tuple[int, float, float, float] | None:
            if not bool(item.get("recovered_current_container_valid")):
                return None
            width = item.get("recovered_current_container_width")
            height = item.get("recovered_current_container_height")
            confidence = item.get("recovered_current_container_confidence")
            if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                return None
            if float(width) <= 0.0 or float(height) <= 0.0:
                return None
            return (
                int(item["index"]),
                float(width),
                float(height),
                float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            )

        for sibling_set in sibling_sets:
            members = [by_index[index] for index in sibling_set if index in by_index]
            for item in members:
                samples = [
                    sample
                    for sibling in members
                    if sibling is not item
                    for sample in [independent_container_sample(sibling)]
                    if sample is not None
                ]
                baseline_valid = len(samples) >= cls.NATURAL_BASELINE_MIN_PEERS
                peer_width = median([sample[1] for sample in samples]) if baseline_valid else None
                peer_height = median([sample[2] for sample in samples]) if baseline_valid else None
                recovered = clipped_box(item.get("recovered_current_container_bbox"))
                semantic = clipped_box(item.get("prepared_bbox"))
                if recovered is not None:
                    center_x = (recovered[0] + recovered[2]) / 2.0
                    center_y = (recovered[1] + recovered[3]) / 2.0
                elif semantic is not None:
                    center_x = (semantic[0] + semantic[2]) / 2.0
                    center_y = (semantic[1] + semantic[3]) / 2.0
                else:
                    center_x = center_y = None
                reference_bbox = (
                    [
                        round(center_x - peer_width / 2.0, 2),
                        round(center_y - peer_height / 2.0, 2),
                        round(center_x + peer_width / 2.0, 2),
                        round(center_y + peer_height / 2.0, 2),
                    ]
                    if baseline_valid
                    and peer_width is not None
                    and peer_height is not None
                    and center_x is not None
                    and center_y is not None
                    else None
                )
                item.update({
                    "independent_natural_baseline_bbox": reference_bbox,
                    "independent_natural_baseline_valid": bool(
                        baseline_valid and reference_bbox is not None
                    ),
                    "independent_natural_baseline_source": (
                        "unavailable" if not baseline_valid or reference_bbox is None
                        else "single_peer_independent_container" if len(samples) == 1
                        else "leave_one_out_independent_container"
                    ),
                    "independent_natural_baseline_confidence": (
                        sum(sample[3] for sample in samples) / max(len(samples), 1)
                        if baseline_valid else 0.0
                    ),
                    "independent_natural_baseline_sibling_indices": [sample[0] for sample in samples],
                    "independent_natural_baseline_sample_count": len(samples),
                    "natural_baseline_peer_width": peer_width,
                    "natural_baseline_peer_height": peer_height,
                    "independent_focus_scale_w": (
                        float(item["recovered_current_container_width"]) / max(peer_width, 1e-6)
                        if bool(item.get("recovered_current_container_valid"))
                        and peer_width is not None else None
                    ),
                    "independent_focus_scale_h": (
                        float(item["recovered_current_container_height"]) / max(peer_height, 1e-6)
                        if bool(item.get("recovered_current_container_valid"))
                        and peer_height is not None else None
                    ),
                })

        def natural_baseline_sample(
            item: dict[str, Any],
        ) -> tuple[dict[str, float], str] | None:
            """Return a reliable peer's semantic-to-container padding sample."""
            if (
                not bool(item.get("extent_reliable"))
                or not bool(item.get("extent_valid"))
                or bool(item.get("extent_has_censored_measurement"))
                or int(item.get("extent_device_boundary_contaminated_side_count", 0)) > 0
            ):
                return None
            semantic = clipped_box(item.get("prepared_bbox"))
            extent = clipped_box(item.get("visual_extent_bbox"))
            if semantic is None or extent is None:
                return None
            left, top, right, bottom = semantic
            width = max(right - left, 1e-6)
            height = max(bottom - top, 1e-6)
            source = (
                "fully_measured"
                if bool(item.get("extent_fully_measured"))
                else "reconstructed"
                if any(
                    str(item.get(field, "")).startswith("mirrored_")
                    for field in (
                        "extent_left_source",
                        "extent_right_source",
                        "extent_top_source",
                        "extent_bottom_source",
                    )
                )
                else ""
            )
            if not source:
                return None
            return ({
                "left": cls._clamp01(max(0.0, (left - extent[0]) / width) /
                    max(cls.NATURAL_BASELINE_MAX_PADDING_RATIO_X, 1e-6))
                * cls.NATURAL_BASELINE_MAX_PADDING_RATIO_X,
                "right": cls._clamp01(max(0.0, (extent[2] - right) / width) /
                    max(cls.NATURAL_BASELINE_MAX_PADDING_RATIO_X, 1e-6))
                * cls.NATURAL_BASELINE_MAX_PADDING_RATIO_X,
                "top": cls._clamp01(max(0.0, (top - extent[1]) / height) /
                    max(cls.NATURAL_BASELINE_MAX_PADDING_RATIO_Y, 1e-6))
                * cls.NATURAL_BASELINE_MAX_PADDING_RATIO_Y,
                "bottom": cls._clamp01(max(0.0, (extent[3] - bottom) / height) /
                    max(cls.NATURAL_BASELINE_MAX_PADDING_RATIO_Y, 1e-6))
                * cls.NATURAL_BASELINE_MAX_PADDING_RATIO_Y,
            }, source)

        def clip_natural_container(
            box: tuple[float, float, float, float],
            item: dict[str, Any],
        ) -> tuple[float, float, float, float] | None:
            left, top, right, bottom = box
            visual_cell = clipped_box(item.get("visual_cell_bbox"))
            tile = clipped_box(item.get("prepared_montage_tile_bbox"))
            for boundary in (visual_cell, tile):
                if boundary is not None:
                    left = max(left, boundary[0])
                    top = max(top, boundary[1])
                    right = min(right, boundary[2])
                    bottom = min(bottom, boundary[3])
            for side, value in (
                ("left", item.get("prepared_device_left")),
                ("top", item.get("prepared_device_top")),
                ("right", item.get("prepared_device_right")),
                ("bottom", item.get("prepared_device_bottom")),
            ):
                if not isinstance(value, (int, float)):
                    continue
                if side == "left":
                    left = max(left, float(value))
                elif side == "top":
                    top = max(top, float(value))
                elif side == "right":
                    right = min(right, float(value))
                else:
                    bottom = min(bottom, float(value))
            return clipped_box((left, top, right, bottom))

        for sibling_set in sibling_sets:
            members = [by_index[index] for index in sibling_set if index in by_index]
            for item in members:
                semantic = clipped_box(item.get("prepared_bbox"))
                extent = clipped_box(item.get("visual_extent_bbox"))
                observed_padding: dict[str, float | None] = {
                    "left": None, "right": None, "top": None, "bottom": None,
                }
                if semantic is not None and extent is not None:
                    left, top, right, bottom = semantic
                    width = max(right - left, 1e-6)
                    height = max(bottom - top, 1e-6)
                    observed_padding = {
                        "left": (left - extent[0]) / width,
                        "right": (extent[2] - right) / width,
                        "top": (top - extent[1]) / height,
                        "bottom": (extent[3] - bottom) / height,
                    }
                samples: list[tuple[int, dict[str, float], str]] = []
                for sibling in members:
                    if sibling is item:
                        continue
                    sample = natural_baseline_sample(sibling)
                    if sample is not None:
                        samples.append((int(sibling["index"]), sample[0], sample[1]))
                fully_measured_samples = [sample for sample in samples if sample[2] == "fully_measured"]
                reconstructed_samples = [sample for sample in samples if sample[2] == "reconstructed"]
                selected_samples = list(fully_measured_samples)
                if len(selected_samples) < cls.NATURAL_BASELINE_MIN_PEERS:
                    selected_samples.extend(reconstructed_samples)
                baseline_padding: dict[str, float | None] = {
                    "left": None, "right": None, "top": None, "bottom": None,
                }
                natural_container = None
                baseline_valid = False
                if semantic is not None and len(selected_samples) >= cls.NATURAL_BASELINE_MIN_PEERS:
                    left, top, right, bottom = semantic
                    width = max(right - left, 1e-6)
                    height = max(bottom - top, 1e-6)
                    baseline_padding = {
                        side: median([sample[1][side] for sample in selected_samples])
                        for side in ("left", "right", "top", "bottom")
                    }
                    natural_container = clip_natural_container((
                        left - float(baseline_padding["left"]) * width,
                        top - float(baseline_padding["top"]) * height,
                        right + float(baseline_padding["right"]) * width,
                        bottom + float(baseline_padding["bottom"]) * height,
                    ), item)
                    baseline_valid = bool(
                        natural_container is not None
                        and natural_container[0] <= left
                        and natural_container[1] <= top
                        and natural_container[2] >= right
                        and natural_container[3] >= bottom
                    )
                source = (
                    "unavailable" if not baseline_valid
                    else "single_peer" if len(selected_samples) == 1
                    else "leave_one_out_fully_measured" if all(
                        sample[2] == "fully_measured" for sample in selected_samples
                    )
                    else "leave_one_out_mixed"
                )
                confidence = (
                    0.0 if not baseline_valid else sum(
                        1.0 if sample[2] == "fully_measured" else 0.75
                        for sample in selected_samples
                    ) / max(len(selected_samples), 1)
                )
                outlier_score = (
                    sum(
                        abs(float(observed_padding[side]) - float(baseline_padding[side]))
                        for side in ("left", "right", "top", "bottom")
                    ) / 4.0
                    if baseline_valid and all(value is not None for value in observed_padding.values())
                    else None
                )
                natural_width = (
                    max(0.0, natural_container[2] - natural_container[0])
                    if natural_container is not None else None
                )
                natural_height = (
                    max(0.0, natural_container[3] - natural_container[1])
                    if natural_container is not None else None
                )
                item.update({
                    "observed_left_padding_ratio": observed_padding["left"],
                    "observed_right_padding_ratio": observed_padding["right"],
                    "observed_top_padding_ratio": observed_padding["top"],
                    "observed_bottom_padding_ratio": observed_padding["bottom"],
                    "baseline_left_padding_ratio": baseline_padding["left"],
                    "baseline_right_padding_ratio": baseline_padding["right"],
                    "baseline_top_padding_ratio": baseline_padding["top"],
                    "baseline_bottom_padding_ratio": baseline_padding["bottom"],
                    "natural_container_bbox": [round(value, 2) for value in natural_container] if natural_container else None,
                    "natural_container_width": natural_width,
                    "natural_container_height": natural_height,
                    "natural_container_area": natural_width * natural_height if natural_width is not None and natural_height is not None else None,
                    "natural_container_baseline_valid": baseline_valid,
                    "natural_container_baseline_confidence": confidence,
                    "natural_container_baseline_source": source,
                    "natural_baseline_sibling_indices": [sample[0] for sample in selected_samples],
                    "natural_baseline_sample_count": len(selected_samples),
                    "natural_baseline_fully_measured_sample_count": sum(
                        sample[2] == "fully_measured" for sample in selected_samples
                    ),
                    "natural_baseline_reconstructed_sample_count": sum(
                        sample[2] == "reconstructed" for sample in selected_samples
                    ),
                    "focus_expansion_w": (
                        float(item.get("visual_extent_width", 0.0)) / max(natural_width, 1e-6)
                        if baseline_valid and natural_width is not None else None
                    ),
                    "focus_expansion_h": (
                        float(item.get("visual_extent_height", 0.0)) / max(natural_height, 1e-6)
                        if baseline_valid and natural_height is not None else None
                    ),
                    "semantic_container_padding_outlier_score": outlier_score,
                })

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
                relative_extent_expansion_width = (
                    expansion_width / max(median_expansion_width, 1e-6)
                )
                relative_extent_expansion_height = (
                    expansion_height / max(median_expansion_height, 1e-6)
                )
                relative_visual_width = width / max(base_width, 1e-6)
                relative_visual_height = height / max(base_height, 1e-6)
                relative_visual_area = area / max(base_area, 1e-6)
                width_consistency = math.exp(-abs(math.log(max(expansion_width, 1e-6) / max(median_expansion_width, 1e-6))) / 0.45)
                height_consistency = math.exp(-abs(math.log(max(expansion_height, 1e-6) / max(median_expansion_height, 1e-6))) / 0.45)
                aspect_consistency = math.exp(-abs(math.log(max(relative_visual_width / max(relative_visual_height, 1e-6), 1e-6))) / cls.SCALE_BALANCE_SIGMA)
                footprint_consistency = cls._clamp01((width_consistency * height_consistency * aspect_consistency) ** (1.0 / 3.0))
                footprint_valid = footprint_consistency >= 0.20
                uniform_scale = math.sqrt(max(relative_extent_expansion_width * relative_extent_expansion_height, 0.0))
                scale_growth = max(0.0, uniform_scale - 1.0)
                width_growth = max(0.0, relative_extent_expansion_width - 1.0)
                height_growth = max(0.0, relative_extent_expansion_height - 1.0)
                uniform_growth = min(width_growth, height_growth)
                scale_balance = math.exp(-abs(math.log(max(relative_extent_expansion_width / max(relative_extent_expansion_height, 1e-6), 1e-6))) / cls.SCALE_BALANCE_SIGMA)
                base_score = cls._clamp01(scale_growth / max(cls.ENLARGEMENT_FULL_SCALE_GROWTH, 1e-6))
                balance_gate = cls.ENLARGEMENT_BALANCE_FLOOR + (1.0 - cls.ENLARGEMENT_BALANCE_FLOOR) * scale_balance
                two_axis_support = cls._clamp01(uniform_growth / max(cls.ENLARGEMENT_MIN_MEANINGFUL_GROWTH, 1e-6))
                footprint_gate = 0.5 + 0.5 * footprint_consistency
                extent_symmetry = float(item.get("extent_symmetry", 0.0))
                extent_symmetry_gate = cls.ENLARGEMENT_SYMMETRY_FLOOR + (1.0 - cls.ENLARGEMENT_SYMMETRY_FLOOR) * extent_symmetry
                extent_valid = (
                    bool(item.get("extent_reliable", False))
                    and float(item.get("visual_extent_width", 0.0)) > 0.0
                    and float(item.get("visual_extent_height", 0.0)) > 0.0
                )
                debug_semantic_metrics = semantic_metrics(item)
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
                    "relative_extent_expansion_width": relative_extent_expansion_width,
                    "relative_extent_expansion_height": relative_extent_expansion_height,
                    "relative_extent_expansion_w": relative_extent_expansion_width,
                    "relative_extent_expansion_h": relative_extent_expansion_height,
                    "candidate_expansion_w": expansion_width,
                    "candidate_expansion_h": expansion_height,
                    "median_expansion_w": median_expansion_width,
                    "median_expansion_h": median_expansion_height,
                    "enlargement_median_sample_count": len(members),
                    "prepared_candidate_width": debug_semantic_metrics[0] if debug_semantic_metrics else None,
                    "prepared_candidate_height": debug_semantic_metrics[1] if debug_semantic_metrics else None,
                    "enlargement_semantic_width": debug_semantic_metrics[0] if debug_semantic_metrics else None,
                    "enlargement_semantic_height": debug_semantic_metrics[1] if debug_semantic_metrics else None,
                    "extent_uniform_scale": uniform_scale,
                    "extent_scale_growth": scale_growth,
                    "extent_scale_balance": scale_balance,
                    "extent_scale_balance_gate": balance_gate,
                    "extent_two_axis_support": two_axis_support,
                    "extent_symmetry_gate": extent_symmetry_gate,
                    "width_growth": width_growth,
                    "height_growth": height_growth,
                    "uniform_growth": uniform_growth,
                    "scale_balance": scale_balance,
                    "two_axis_support": two_axis_support,
                    "base_score": base_score,
                    "balance_gate": balance_gate,
                    "extent_valid": extent_valid,
                    "sibling_median_footprint_width_ratio": median_expansion_width,
                    "sibling_median_footprint_height_ratio": median_expansion_height,
                    "sibling_median_extent_width_ratio": median_expansion_width,
                    "sibling_median_extent_height_ratio": median_expansion_height,
                    "footprint_to_semantic_width_ratio": expansion_width,
                    "footprint_to_semantic_height_ratio": expansion_height,
                    "footprint_consistency_score": footprint_consistency,
                    "footprint_valid": footprint_valid,
                    "enlargement_score": 0.0 if not extent_valid else cls._clamp01(base_score * balance_gate * two_axis_support * extent_symmetry_gate),
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
    def _save_cv_prepared_debug_artifacts(
        cls,
        image: Image.Image,
        candidates: list[dict[str, Any]],
        prepared_candidate_bboxes: dict[int, list[int]],
        candidate_groups: list[list[int]],
        peer_analysis: dict[str, Any],
        sibling_analysis: dict[str, Any],
        evidence: list[dict[str, Any]],
        focus_image_mode: str,
        montage_grid: Any,
        montage_size: Any,
        montage_tile_sizes: Any,
        roi_bbox: Any,
        source_device_geometry: dict[str, Any] | None = None,
    ) -> None:
        """Save a raw prepared CV image and a non-invasive geometry overlay."""
        raw = image.convert("RGB").copy()
        debug_image = raw.copy()
        draw = ImageDraw.Draw(debug_image)
        evidence_by_index = {int(item["index"]): item for item in evidence if isinstance(item, dict) and "index" in item}
        peer_by_index = peer_analysis.get("peer_group_by_index", {})
        sibling_by_index = sibling_analysis.get("sibling_group_by_index", {})

        def draw_box(box: Any, color: tuple[int, int, int], width: int = 2) -> None:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                draw.rectangle(tuple(int(round(float(value))) for value in box[:4]), outline=color, width=width)

        draw.rectangle((0, 0, min(raw.width, 620), 58), fill=(0, 0, 0))
        mode_line = f"MODE: {focus_image_mode}  PREPARED: {raw.width}x{raw.height}"
        draw.text((8, 6), mode_line, fill=(255, 255, 255))
        if focus_image_mode == "group_montage":
            draw.text((8, 24), f"GRID: {montage_grid}  TILE: group boundaries", fill=(255, 255, 255))
        elif focus_image_mode == "roi":
            draw.text((8, 24), f"ROI: {roi_bbox}  INPUT: {raw.width}x{raw.height}", fill=(255, 255, 255))
        else:
            draw.text((8, 24), "FULL IMAGE", fill=(255, 255, 255))
        draw.text((8, 42), "BOX=semantic IC=current container NAT=legacy baseline CELL=ownership CARD=card cell OBS=card window EXT=visual extent", fill=(255, 255, 255))

        montage_tile_bboxes = []
        if focus_image_mode == "group_montage" and isinstance(montage_tile_sizes, list):
            montage_tile_bboxes = [
                tile for tile in montage_tile_sizes
                if isinstance(tile, dict) and isinstance(tile.get("bbox"), (list, tuple))
            ]
            for tile in montage_tile_bboxes:
                draw_box(tile.get("bbox"), (255, 0, 255), 3)
                bbox = tile.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    draw.text((int(bbox[0]) + 4, int(bbox[1]) + 4), f"TILE {tile.get('tile_index', '?')} / G{tile.get('group_id', '?')}", fill=(255, 0, 255))

        device_edges: dict[str, set[float]] = {
            "left": set(), "right": set(), "top": set(), "bottom": set(),
        }
        for item in evidence_by_index.values():
            for side in device_edges:
                value = item.get(f"prepared_device_{side}")
                if isinstance(value, (int, float)):
                    device_edges[side].add(float(value))
        for side, values in device_edges.items():
            for value in values:
                if side in ("left", "right"):
                    x = int(round(value))
                    draw.line((x, 0, x, raw.height - 1), fill=(255, 80, 80), width=1)
                    draw.text((x + 2, 60), f"DEVICE {side[0].upper()}", fill=(255, 80, 80))
                else:
                    y = int(round(value))
                    draw.line((0, y, raw.width - 1, y), fill=(255, 80, 80), width=1)
                    draw.text((8, y + 2), f"DEVICE {side[0].upper()}", fill=(255, 80, 80))

        for index, bbox in sorted(prepared_candidate_bboxes.items()):
            item = evidence_by_index.get(int(index), {})
            peer_id = peer_by_index.get(int(index), -1)
            sibling_id = sibling_by_index.get(int(index), -1)
            draw_box(bbox, (255, 255, 0), 2)
            draw_box(item.get("recovered_current_container_bbox"), (0, 220, 210), 2)
            draw_box(item.get("natural_container_bbox"), (190, 120, 255), 2)
            draw_box(item.get("visual_cell_bbox"), (0, 180, 255), 2)
            draw_box(item.get("enlargement_card_cell_bbox"), (255, 0, 255), 2)
            draw_box(item.get("enlargement_card_observation_bbox"), (255, 140, 0), 2)
            draw_box(item.get("visual_extent_bbox"), (80, 255, 100), 3)
            mirrored_extent = any(
                str(item.get(field, "")).startswith("mirrored_")
                for field in (
                    "extent_left_source",
                    "extent_right_source",
                    "extent_top_source",
                    "extent_bottom_source",
                )
            )
            extent_state = (
                "C" if item.get("extent_reliable") and item.get("extent_has_censored_measurement")
                else "R" if item.get("extent_reliable") and not mirrored_extent
                else "M" if item.get("extent_reliable") and mirrored_extent
                else "T" if item.get("extent_truncated")
                else "-"
            )
            hit_sides = "".join(
                marker
                for side_marker, hit_field, state_field in (
                    ("L", "extent_obs_hit_left", "extent_left_state"),
                    ("R", "extent_obs_hit_right", "extent_right_state"),
                    ("T", "extent_obs_hit_top", "extent_top_state"),
                    ("B", "extent_obs_hit_bottom", "extent_bottom_state"),
                )
                if item.get(hit_field)
                for marker in (
                    f"{side_marker}~"
                    if item.get(state_field) == "continuation_to_limit"
                    else f"{side_marker}!"
                )
            )
            device_sides = "".join(
                marker
                for marker, field in (
                    ("LD", "extent_left_device_boundary_contaminated"),
                    ("RD", "extent_right_device_boundary_contaminated"),
                    ("TD", "extent_top_device_boundary_contaminated"),
                    ("BD", "extent_bottom_device_boundary_contaminated"),
                ) if item.get(field)
            )
            label = f"#{index} P{peer_id}/S{sibling_id if sibling_id >= 0 else '-'} EXT:{extent_state} {hit_sides} {device_sides}".rstrip()
            try:
                left = int(round(float(bbox[0])))
                top = max(62, int(round(float(bbox[1]))) - 16)
                draw.rectangle((left, top, left + max(90, len(label) * 7), top + 15), fill=(0, 0, 0))
                draw.text((left + 2, top + 1), label, fill=(255, 255, 255))
            except (TypeError, ValueError):
                pass

        raw.save(cls.DEBUG_CV_PREPARED_IMAGE_PATH, format="JPEG", quality=95)
        debug_image.save(cls.DEBUG_CV_PREPARED_DEBUG_IMAGE_PATH, format="JPEG", quality=95)

        candidate_metadata = []
        text_by_index = {index: str(candidate.get("text") or "") for index, candidate in enumerate(candidates) if isinstance(candidate, dict)}
        for index, bbox in sorted(prepared_candidate_bboxes.items()):
            item = evidence_by_index.get(int(index), {})
            candidate_metadata.append({
                "index": int(index),
                "text": text_by_index.get(int(index), ""),
                "prepared_bbox": bbox,
                "prepared_candidate_width": item.get("prepared_candidate_width"),
                "prepared_candidate_height": item.get("prepared_candidate_height"),
                "prepared_montage_tile_bbox": item.get("prepared_montage_tile_bbox"),
                "source_device_viewport_bbox": item.get("source_device_viewport_bbox"),
                "source_device_viewport_valid": item.get("source_device_viewport_valid"),
                "source_device_left_valid": item.get("source_device_left_valid"),
                "source_device_right_valid": item.get("source_device_right_valid"),
                "source_device_top_valid": item.get("source_device_top_valid"),
                "source_device_bottom_valid": item.get("source_device_bottom_valid"),
                "source_device_left_confidence": item.get("source_device_left_confidence"),
                "source_device_right_confidence": item.get("source_device_right_confidence"),
                "source_device_top_confidence": item.get("source_device_top_confidence"),
                "source_device_bottom_confidence": item.get("source_device_bottom_confidence"),
                "enlargement_semantic_width": item.get("enlargement_semantic_width"),
                "enlargement_semantic_height": item.get("enlargement_semantic_height"),
                "visual_cell_bbox": item.get("visual_cell_bbox"),
                "enlargement_local_observation_bbox": item.get("enlargement_local_observation_bbox"),
                "enlargement_card_cell_bbox": item.get("enlargement_card_cell_bbox"),
                "enlargement_card_observation_bbox": item.get("enlargement_card_observation_bbox"),
                "enlargement_card_observation_valid": item.get("enlargement_card_observation_valid"),
                "enlargement_extent_observation_source": item.get("enlargement_extent_observation_source"),
                "prepared_device_viewport_bbox": item.get("prepared_device_viewport_bbox"),
                "prepared_device_left": item.get("prepared_device_left"),
                "prepared_device_right": item.get("prepared_device_right"),
                "prepared_device_top": item.get("prepared_device_top"),
                "prepared_device_bottom": item.get("prepared_device_bottom"),
                "prepared_device_geometry_source": item.get("prepared_device_geometry_source"),
                "device_geometry_transform_mode": item.get("device_geometry_transform_mode"),
                "visual_extent_bbox": item.get("visual_extent_bbox"),
                "visual_extent_width": item.get("visual_extent_width"),
                "visual_extent_height": item.get("visual_extent_height"),
                "visual_extent_area": item.get("visual_extent_area"),
                "recovered_current_container_bbox": item.get("recovered_current_container_bbox"),
                "recovered_current_container_width": item.get("recovered_current_container_width"),
                "recovered_current_container_height": item.get("recovered_current_container_height"),
                "recovered_current_container_area": item.get("recovered_current_container_area"),
                "recovered_current_container_valid": item.get("recovered_current_container_valid"),
                "recovered_current_container_confidence": item.get("recovered_current_container_confidence"),
                "recovered_current_container_source": item.get("recovered_current_container_source"),
                "recovered_container_left": item.get("recovered_container_left"),
                "recovered_container_right": item.get("recovered_container_right"),
                "recovered_container_top": item.get("recovered_container_top"),
                "recovered_container_bottom": item.get("recovered_container_bottom"),
                "natural_container_left_state": item.get("natural_container_left_state"),
                "natural_container_right_state": item.get("natural_container_right_state"),
                "natural_container_top_state": item.get("natural_container_top_state"),
                "natural_container_bottom_state": item.get("natural_container_bottom_state"),
                "natural_container_left_reason": item.get("natural_container_left_reason"),
                "natural_container_right_reason": item.get("natural_container_right_reason"),
                "natural_container_top_reason": item.get("natural_container_top_reason"),
                "natural_container_bottom_reason": item.get("natural_container_bottom_reason"),
                "natural_container_left_score": item.get("natural_container_left_score"),
                "natural_container_right_score": item.get("natural_container_right_score"),
                "natural_container_top_score": item.get("natural_container_top_score"),
                "natural_container_bottom_score": item.get("natural_container_bottom_score"),
                "natural_container_left_distance": item.get("natural_container_left_distance"),
                "natural_container_right_distance": item.get("natural_container_right_distance"),
                "natural_container_top_distance": item.get("natural_container_top_distance"),
                "natural_container_bottom_distance": item.get("natural_container_bottom_distance"),
                "natural_container_left_scan_mode": item.get("natural_container_left_scan_mode"),
                "natural_container_right_scan_mode": item.get("natural_container_right_scan_mode"),
                "natural_container_top_scan_mode": item.get("natural_container_top_scan_mode"),
                "natural_container_bottom_scan_mode": item.get("natural_container_bottom_scan_mode"),
                "natural_container_left_selected_scan_step": item.get("natural_container_left_selected_scan_step"),
                "natural_container_right_selected_scan_step": item.get("natural_container_right_selected_scan_step"),
                "natural_container_top_selected_scan_step": item.get("natural_container_top_selected_scan_step"),
                "natural_container_bottom_selected_scan_step": item.get("natural_container_bottom_selected_scan_step"),
                "natural_container_left_span_support": item.get("natural_container_left_span_support"),
                "natural_container_right_span_support": item.get("natural_container_right_span_support"),
                "natural_container_top_span_support": item.get("natural_container_top_span_support"),
                "natural_container_bottom_span_support": item.get("natural_container_bottom_span_support"),
                "natural_container_left_mean_oriented_strength": item.get("natural_container_left_mean_oriented_strength"),
                "natural_container_right_mean_oriented_strength": item.get("natural_container_right_mean_oriented_strength"),
                "natural_container_top_mean_oriented_strength": item.get("natural_container_top_mean_oriented_strength"),
                "natural_container_bottom_mean_oriented_strength": item.get("natural_container_bottom_mean_oriented_strength"),
                "natural_container_left_inside_outside_contrast": item.get("natural_container_left_inside_outside_contrast"),
                "natural_container_right_inside_outside_contrast": item.get("natural_container_right_inside_outside_contrast"),
                "natural_container_top_inside_outside_contrast": item.get("natural_container_top_inside_outside_contrast"),
                "natural_container_bottom_inside_outside_contrast": item.get("natural_container_bottom_inside_outside_contrast"),
                "natural_container_dense_boundary_side_count": item.get("natural_container_dense_boundary_side_count"),
                "recovered_container_left_padding_ratio": item.get("recovered_container_left_padding_ratio"),
                "recovered_container_right_padding_ratio": item.get("recovered_container_right_padding_ratio"),
                "recovered_container_top_padding_ratio": item.get("recovered_container_top_padding_ratio"),
                "recovered_container_bottom_padding_ratio": item.get("recovered_container_bottom_padding_ratio"),
                "semantic_coverage_w": item.get("semantic_coverage_w"),
                "semantic_coverage_h": item.get("semantic_coverage_h"),
                "independent_natural_baseline_bbox": item.get("independent_natural_baseline_bbox"),
                "independent_natural_baseline_valid": item.get("independent_natural_baseline_valid"),
                "independent_natural_baseline_source": item.get("independent_natural_baseline_source"),
                "independent_natural_baseline_confidence": item.get("independent_natural_baseline_confidence"),
                "independent_natural_baseline_sibling_indices": item.get("independent_natural_baseline_sibling_indices"),
                "independent_natural_baseline_sample_count": item.get("independent_natural_baseline_sample_count"),
                "natural_baseline_peer_width": item.get("natural_baseline_peer_width"),
                "natural_baseline_peer_height": item.get("natural_baseline_peer_height"),
                "independent_focus_scale_w": item.get("independent_focus_scale_w"),
                "independent_focus_scale_h": item.get("independent_focus_scale_h"),
                "natural_container_bbox": item.get("natural_container_bbox"),
                "natural_container_width": item.get("natural_container_width"),
                "natural_container_height": item.get("natural_container_height"),
                "natural_container_area": item.get("natural_container_area"),
                "natural_container_baseline_valid": item.get("natural_container_baseline_valid"),
                "natural_container_baseline_confidence": item.get("natural_container_baseline_confidence"),
                "natural_container_baseline_source": item.get("natural_container_baseline_source"),
                "natural_baseline_sibling_indices": item.get("natural_baseline_sibling_indices"),
                "natural_baseline_sample_count": item.get("natural_baseline_sample_count"),
                "natural_baseline_fully_measured_sample_count": item.get("natural_baseline_fully_measured_sample_count"),
                "natural_baseline_reconstructed_sample_count": item.get("natural_baseline_reconstructed_sample_count"),
                "observed_left_padding_ratio": item.get("observed_left_padding_ratio"),
                "observed_right_padding_ratio": item.get("observed_right_padding_ratio"),
                "observed_top_padding_ratio": item.get("observed_top_padding_ratio"),
                "observed_bottom_padding_ratio": item.get("observed_bottom_padding_ratio"),
                "baseline_left_padding_ratio": item.get("baseline_left_padding_ratio"),
                "baseline_right_padding_ratio": item.get("baseline_right_padding_ratio"),
                "baseline_top_padding_ratio": item.get("baseline_top_padding_ratio"),
                "baseline_bottom_padding_ratio": item.get("baseline_bottom_padding_ratio"),
                "focus_expansion_w": item.get("focus_expansion_w"),
                "focus_expansion_h": item.get("focus_expansion_h"),
                "semantic_container_padding_outlier_score": item.get("semantic_container_padding_outlier_score"),
                "enlargement_sibling_group_id": item.get("enlargement_sibling_group_id"),
                "enlargement_sibling_indices": item.get("enlargement_sibling_indices"),
                "enlargement_sibling_count": item.get("enlargement_sibling_count"),
                "enlargement_median_sample_count": item.get("enlargement_median_sample_count"),
                "candidate_expansion_w": item.get("candidate_expansion_w"),
                "candidate_expansion_h": item.get("candidate_expansion_h"),
                "median_expansion_w": item.get("median_expansion_w"),
                "median_expansion_h": item.get("median_expansion_h"),
                "relative_extent_expansion_width": item.get("relative_extent_expansion_width"),
                "relative_extent_expansion_height": item.get("relative_extent_expansion_height"),
                "relative_extent_expansion_w": item.get("relative_extent_expansion_w"),
                "relative_extent_expansion_h": item.get("relative_extent_expansion_h"),
                "extent_uniform_scale": item.get("extent_uniform_scale"),
                "extent_scale_growth": item.get("extent_scale_growth"),
                "extent_scale_balance": item.get("extent_scale_balance"),
                "extent_scale_balance_gate": item.get("extent_scale_balance_gate"),
                "extent_two_axis_support": item.get("extent_two_axis_support"),
                "width_growth": item.get("width_growth"),
                "height_growth": item.get("height_growth"),
                "uniform_growth": item.get("uniform_growth"),
                "scale_balance": item.get("scale_balance"),
                "two_axis_support": item.get("two_axis_support"),
                "extent_symmetry": item.get("extent_symmetry"),
                "extent_symmetry_gate": item.get("extent_symmetry_gate"),
                "base_score": item.get("base_score"),
                "base_enlargement_score": item.get("base_enlargement_score"),
                "balance_gate": item.get("balance_gate"),
                "enlargement_score": item.get("enlargement_score"),
                "extent_valid": item.get("extent_valid"),
                "enlargement_extent_reason": item.get("enlargement_extent_reason"),
                "extent_reliable": item.get("extent_reliable"),
                "extent_boundary_reliability": item.get("extent_boundary_reliability"),
                "extent_truncated": item.get("extent_truncated"),
                "extent_horizontal_state": item.get("extent_horizontal_state"),
                "extent_vertical_state": item.get("extent_vertical_state"),
                "extent_censored_side_count": item.get("extent_censored_side_count"),
                "extent_has_censored_measurement": item.get("extent_has_censored_measurement"),
                "extent_fully_measured": item.get("extent_fully_measured"),
                "extent_width_is_lower_bound": item.get("extent_width_is_lower_bound"),
                "extent_height_is_lower_bound": item.get("extent_height_is_lower_bound"),
                "extent_area_is_lower_bound": item.get("extent_area_is_lower_bound"),
                "extent_edge_coherent_side_count": item.get("extent_edge_coherent_side_count"),
                "extent_mean_edge_coherence": item.get("extent_mean_edge_coherence"),
                "extent_obs_boundary_hit_count": item.get("extent_obs_boundary_hit_count"),
                "extent_boundary_debug": item.get("extent_boundary_debug"),
                "extent_left_boundary_source": item.get("extent_left_boundary_source"),
                "extent_right_boundary_source": item.get("extent_right_boundary_source"),
                "extent_top_boundary_source": item.get("extent_top_boundary_source"),
                "extent_bottom_boundary_source": item.get("extent_bottom_boundary_source"),
                "extent_left_selected_edge_coordinate": item.get("extent_left_selected_edge_coordinate"),
                "extent_right_selected_edge_coordinate": item.get("extent_right_selected_edge_coordinate"),
                "extent_top_selected_edge_coordinate": item.get("extent_top_selected_edge_coordinate"),
                "extent_bottom_selected_edge_coordinate": item.get("extent_bottom_selected_edge_coordinate"),
                "extent_left_distance_to_device_boundary": item.get("extent_left_distance_to_device_boundary"),
                "extent_right_distance_to_device_boundary": item.get("extent_right_distance_to_device_boundary"),
                "extent_top_distance_to_device_boundary": item.get("extent_top_distance_to_device_boundary"),
                "extent_bottom_distance_to_device_boundary": item.get("extent_bottom_distance_to_device_boundary"),
                "extent_left_device_boundary_contaminated": item.get("extent_left_device_boundary_contaminated"),
                "extent_right_device_boundary_contaminated": item.get("extent_right_device_boundary_contaminated"),
                "extent_top_device_boundary_contaminated": item.get("extent_top_device_boundary_contaminated"),
                "extent_bottom_device_boundary_contaminated": item.get("extent_bottom_device_boundary_contaminated"),
                "extent_device_boundary_contaminated_side_count": item.get("extent_device_boundary_contaminated_side_count"),
            })
        metadata = {
            "image_name": None,
            "focus_image_mode": focus_image_mode,
            "prepared_size": [raw.width, raw.height],
            "roi_bbox_pixels": roi_bbox,
            "source_device_viewport": source_device_geometry or {},
            "montage_grid": montage_grid,
            "montage_size": montage_size,
            "montage_tile_sizes": montage_tile_sizes,
            "montage_tile_bboxes": montage_tile_bboxes,
            "prepared_candidate_bboxes": {str(index): bbox for index, bbox in prepared_candidate_bboxes.items()},
            "peer_groups": peer_analysis.get("peer_sets", []),
            "isolated_indices": peer_analysis.get("isolated_indices", []),
            "enlargement_sibling_groups": sibling_analysis.get("sibling_sets", []),
            "candidates": candidate_metadata,
        }
        with open(cls.DEBUG_CV_PREPARED_METADATA_PATH, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def _save_cv_final_debug_image(
        cls,
        image: Image.Image,
        candidates: list[dict[str, Any]],
        prepared_candidate_bboxes: dict[int, list[int]],
        evidence: list[dict[str, Any]],
        v5_cascade: dict[str, Any],
    ) -> None:
        """Render final CV diagnostics without deriving any new evidence."""
        debug_image = image.convert("RGB").copy()
        draw = ImageDraw.Draw(debug_image)
        evidence_by_index = {
            int(item["index"]): item
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        }
        text_by_index = {
            index: str(candidate.get("text") or "")
            for index, candidate in enumerate(candidates)
            if isinstance(candidate, dict)
        }

        def draw_box(box: Any, color: tuple[int, int, int], width: int) -> None:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                return
            try:
                draw.rectangle(
                    tuple(int(round(float(value))) for value in box[:4]),
                    outline=color,
                    width=width,
                )
            except (TypeError, ValueError):
                return

        def stage_status(decision: Any) -> str:
            if not isinstance(decision, dict):
                return "ABSTAIN"
            if not bool(decision.get("executed", True)):
                return "SKIPPED"
            if bool(decision.get("matched")):
                return "MATCH"
            return "ABSTAIN" if decision.get("candidate_index") is None else "NO_HIT"

        def score_text(item: dict[str, Any], field: str) -> str:
            try:
                return f"{float(item.get(field, 0.0)):.2f}"
            except (TypeError, ValueError):
                return "n/a"

        stage_lines = [
            f"OUTLINE: {stage_status(v5_cascade.get('outline_decision'))}",
            f"ENLARGEMENT: {stage_status(v5_cascade.get('enlargement_decision'))}",
            f"HIGHLIGHT: {stage_status(v5_cascade.get('highlight_decision'))}",
        ]
        header_height = 14 * (len(stage_lines) + 1) + 8
        draw.rectangle((0, 0, min(debug_image.width, 420), header_height), fill=(0, 0, 0))
        draw.text((8, 5), "CV FINAL  BOX=semantic IC=current container NAT=peer baseline EXT=V5.4 extent", fill=(255, 255, 255))
        for line_index, line in enumerate(stage_lines, start=1):
            draw.text((8, 5 + 14 * line_index), line, fill=(255, 255, 255))

        device_edges: dict[str, set[float]] = {
            "left": set(), "right": set(), "top": set(), "bottom": set(),
        }
        for item in evidence_by_index.values():
            for side in device_edges:
                value = item.get(f"prepared_device_{side}")
                if isinstance(value, (int, float)):
                    device_edges[side].add(float(value))
        for side, values in device_edges.items():
            for value in values:
                if side in ("left", "right"):
                    coordinate = int(round(value))
                    draw.line(
                        (coordinate, 0, coordinate, debug_image.height - 1),
                        fill=(255, 80, 80),
                        width=1,
                    )
                    draw.text((coordinate + 2, header_height + 2), f"DEVICE {side[0].upper()}", fill=(255, 80, 80))
                else:
                    coordinate = int(round(value))
                    draw.line(
                        (0, coordinate, debug_image.width - 1, coordinate),
                        fill=(255, 80, 80),
                        width=1,
                    )
                    draw.text((8, coordinate + 2), f"DEVICE {side[0].upper()}", fill=(255, 80, 80))

        for index, bbox in sorted(prepared_candidate_bboxes.items()):
            item = evidence_by_index.get(int(index), {})
            draw_box(bbox, (255, 255, 0), 2)
            if bool(item.get("recovered_current_container_valid")):
                draw_box(item.get("recovered_current_container_bbox"), (0, 220, 210), 2)
            if bool(item.get("independent_natural_baseline_valid")):
                draw_box(item.get("independent_natural_baseline_bbox"), (190, 120, 255), 2)
            if bool(item.get("extent_valid")) and bool(item.get("extent_reliable")):
                draw_box(item.get("visual_extent_bbox"), (80, 255, 100), 3)
            device_markers = " ".join(
                marker
                for marker, field in (
                    ("LD", "extent_left_device_boundary_contaminated"),
                    ("RD", "extent_right_device_boundary_contaminated"),
                    ("TD", "extent_top_device_boundary_contaminated"),
                    ("BD", "extent_bottom_device_boundary_contaminated"),
                )
                if bool(item.get(field))
            )
            text = text_by_index.get(int(index), "")
            if len(text) > 28:
                text = f"{text[:25]}..."
            label = (
                f"#{index} {text}\n"
                f"O:{score_text(item, 'outline_score')} "
                f"E:{score_text(item, 'enlargement_score')} "
                f"H:{score_text(item, 'highlight_score')}"
            )
            if device_markers:
                label = f"{label}\n{device_markers}"
            try:
                left = int(round(float(bbox[0])))
                top = max(header_height + 3, int(round(float(bbox[1]))) - 34)
            except (TypeError, ValueError):
                continue
            lines = label.splitlines()
            label_width = max(
                (draw.textbbox((0, 0), line)[2] for line in lines),
                default=0,
            )
            label_height = 13 * len(lines) + 4
            draw.rectangle(
                (left, top, left + label_width + 6, top + label_height),
                fill=(0, 0, 0),
            )
            for line_index, line in enumerate(lines):
                draw.text((left + 3, top + 2 + 13 * line_index), line, fill=(255, 255, 255))

        debug_image.save(cls.DEBUG_CV_FINAL_IMAGE_PATH, format="JPEG", quality=95)

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
    def _estimate_source_device_viewport(cls, image: Image.Image) -> dict[str, Any]:
        """Estimate only globally coherent display edges in source coordinates."""
        pixels = image.convert("RGB")
        width, height = pixels.size
        if width <= 2 or height <= 2:
            return {
                "source_device_viewport_bbox": [None, None, None, None],
                "source_device_viewport_valid": False,
                "source_device_left_valid": False,
                "source_device_right_valid": False,
                "source_device_top_valid": False,
                "source_device_bottom_valid": False,
                "source_device_left_confidence": 0.0,
                "source_device_right_confidence": 0.0,
                "source_device_top_confidence": 0.0,
                "source_device_bottom_confidence": 0.0,
                "source_device_viewport_confidence": 0.0,
                "source_device_geometry_source": "global_long_span_edge_v5_4_2",
            }
        data = pixels.load()
        radius = cls.DEVICE_BOUNDARY_GRADIENT_RADIUS

        def pixel_at(x: int, y: int) -> tuple[int, int, int]:
            return data[
                max(0, min(width - 1, x)),
                max(0, min(height - 1, y)),
            ]

        def luma(color: tuple[int, int, int]) -> float:
            return (
                0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
            ) / 255.0

        def strength(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
            luminance_gradient = abs(luma(first) - luma(second))
            color_gradient = math.sqrt(sum(
                ((first[channel] - second[channel]) / 255.0) ** 2
                for channel in range(3)
            ) / 3.0)
            return cls._clamp01(
                0.45 * cls._clamp01(
                    luminance_gradient
                    / cls.ENLARGEMENT_EDGE_COHERENCE_LUMINANCE_NORMALIZER
                )
                + 0.55 * cls._clamp01(
                    color_gradient
                    / cls.ENLARGEMENT_EDGE_COHERENCE_COLOR_NORMALIZER
                )
            )

        def profile(side: str, coordinate: int) -> tuple[float, float]:
            span = height if side in ("left", "right") else width
            sample_count = max(20, min(160, int(round(span / 16.0))))
            strengths: list[float] = []
            for sample_index in range(sample_count):
                fraction = 0.05 + 0.90 * (sample_index + 0.5) / sample_count
                if side in ("left", "right"):
                    x = coordinate
                    y = int(round(fraction * (height - 1)))
                    normal = strength(pixel_at(x - radius, y), pixel_at(x + radius, y))
                    tangent = strength(pixel_at(x, y - radius), pixel_at(x, y + radius))
                else:
                    x = int(round(fraction * (width - 1)))
                    y = coordinate
                    normal = strength(pixel_at(x, y - radius), pixel_at(x, y + radius))
                    tangent = strength(pixel_at(x - radius, y), pixel_at(x + radius, y))
                strengths.append(normal * normal / max(normal + tangent, 1e-6))
            span_support = sum(
                value >= cls.DEVICE_BOUNDARY_MIN_SAMPLE_STRENGTH
                for value in strengths
            ) / max(len(strengths), 1)
            mean_strength = sum(strengths) / max(len(strengths), 1)
            return span_support, cls._clamp01(
                0.70 * span_support + 0.30 * mean_strength
            )

        def find_edge(side: str) -> tuple[int | None, float]:
            maximum = width if side in ("left", "right") else height
            search = max(radius, int(round(maximum * cls.DEVICE_BOUNDARY_SEARCH_FRACTION)))
            positions = (
                range(radius, min(search, maximum - radius) + 1)
                if side in ("left", "top")
                else range(max(radius, maximum - search), maximum - radius + 1)
            )
            best: tuple[float, int] | None = None
            for position in positions:
                span_support, score = profile(side, position)
                if span_support < cls.DEVICE_BOUNDARY_MIN_SPAN_FRACTION:
                    continue
                if best is None or score > best[0]:
                    best = (score, position)
            if best is None or best[0] < cls.DEVICE_BOUNDARY_MIN_SCORE:
                return None, 0.0
            return best[1], best[0]

        left, left_confidence = find_edge("left")
        right, right_confidence = find_edge("right")
        top, top_confidence = find_edge("top")
        bottom, bottom_confidence = find_edge("bottom")
        horizontal_valid = (
            left is not None and right is not None and right > left
            and right - left >= 0.30 * width
        )
        vertical_valid = (
            top is not None and bottom is not None and bottom > top
            and bottom - top >= 0.30 * height
        )
        confidences = [
            value for value in (
                left_confidence if left is not None else None,
                right_confidence if right is not None else None,
                top_confidence if top is not None else None,
                bottom_confidence if bottom is not None else None,
            ) if value is not None
        ]
        return {
            "source_device_viewport_bbox": [left, top, right, bottom],
            "source_device_viewport_valid": horizontal_valid or vertical_valid,
            "source_device_left_valid": left is not None,
            "source_device_right_valid": right is not None,
            "source_device_top_valid": top is not None,
            "source_device_bottom_valid": bottom is not None,
            "source_device_left_confidence": left_confidence,
            "source_device_right_confidence": right_confidence,
            "source_device_top_confidence": top_confidence,
            "source_device_bottom_confidence": bottom_confidence,
            "source_device_viewport_confidence": (
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
            "source_device_geometry_source": "global_long_span_edge_v5_4_2",
        }

    @classmethod
    def _map_source_device_viewport_to_prepared(
        cls,
        source_geometry: dict[str, Any],
        source_size: tuple[int, int],
        prepared_size: tuple[int, int],
        focus_image_mode: str,
        roi_bbox: tuple[int, int, int, int] | None,
        montage_tiles: Any,
    ) -> dict[int, dict[str, Any]]:
        """Map source device edges only through the actual crop/tile transform."""
        source_bbox = source_geometry.get("source_device_viewport_bbox", [])
        if not isinstance(source_bbox, (list, tuple)) or len(source_bbox) < 4:
            return {}

        def map_region(
            source_region: tuple[float, float, float, float],
            prepared_region: tuple[float, float, float, float],
            indices: list[int],
            transform_mode: str,
        ) -> dict[int, dict[str, Any]]:
            source_left, source_top, source_right, source_bottom = source_region
            prepared_left, prepared_top, prepared_right, prepared_bottom = prepared_region
            scale_x = (prepared_right - prepared_left) / max(source_right - source_left, 1e-6)
            scale_y = (prepared_bottom - prepared_top) / max(source_bottom - source_top, 1e-6)

            def map_edge(value: Any, valid: bool, axis: str) -> float | None:
                if not valid or not isinstance(value, (int, float)):
                    return None
                lower, upper = (
                    (source_left, source_right)
                    if axis == "x" else (source_top, source_bottom)
                )
                if value < lower or value > upper:
                    return None
                if axis == "x":
                    return prepared_left + (float(value) - source_left) * scale_x
                return prepared_top + (float(value) - source_top) * scale_y

            mapped = {
                "prepared_device_left": map_edge(
                    source_bbox[0], bool(source_geometry.get("source_device_left_valid")), "x"
                ),
                "prepared_device_top": map_edge(
                    source_bbox[1], bool(source_geometry.get("source_device_top_valid")), "y"
                ),
                "prepared_device_right": map_edge(
                    source_bbox[2], bool(source_geometry.get("source_device_right_valid")), "x"
                ),
                "prepared_device_bottom": map_edge(
                    source_bbox[3], bool(source_geometry.get("source_device_bottom_valid")), "y"
                ),
                "prepared_device_geometry_source": source_geometry.get("source_device_geometry_source"),
                "device_geometry_transform_mode": transform_mode,
            }
            mapped["prepared_device_viewport_bbox"] = [
                mapped["prepared_device_left"],
                mapped["prepared_device_top"],
                mapped["prepared_device_right"],
                mapped["prepared_device_bottom"],
            ]
            return {index: dict(mapped) for index in indices}

        if focus_image_mode == "group_montage":
            mapped: dict[int, dict[str, Any]] = {}
            if not isinstance(montage_tiles, list):
                return mapped
            for tile in montage_tiles:
                if not isinstance(tile, dict):
                    continue
                source_region = tile.get("source_crop_bbox")
                prepared_region = tile.get("bbox")
                indices = tile.get("candidate_indices")
                if (
                    not isinstance(source_region, (list, tuple))
                    or not isinstance(prepared_region, (list, tuple))
                    or not isinstance(indices, list)
                    or len(source_region) < 4
                    or len(prepared_region) < 4
                ):
                    continue
                mapped.update(map_region(
                    tuple(float(value) for value in source_region[:4]),
                    tuple(float(value) for value in prepared_region[:4]),
                    [int(index) for index in indices],
                    "montage",
                ))
            return mapped

        if focus_image_mode == "roi" and roi_bbox is not None:
            source_region = tuple(float(value) for value in roi_bbox)
            transform_mode = "roi"
        else:
            source_region = (0.0, 0.0, float(source_size[0]), float(source_size[1]))
            transform_mode = "full_image"
        global_mapping = map_region(
            source_region,
            (0.0, 0.0, float(prepared_size[0]), float(prepared_size[1])),
            [0],
            transform_mode,
        )
        return {-1: global_mapping[0]}

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
        list[dict[str, Any]],
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
        list[dict[str, Any]],
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
            tile_sizes: list[dict[str, Any]] = []
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
                    tile_sizes.append({
                        "tile_index": tile_index,
                        "group_id": tile_data["group_id"],
                        "candidate_indices": list(tile_data["indices"]),
                        "bbox": [x, y, x + tile_width, y + tile_height],
                        "source_crop_bbox": [
                            tile_data["crop_left"],
                            tile_data["crop_top"],
                            tile_data["crop_left"] + tile_data["crop"].width,
                            tile_data["crop_top"] + tile_data["crop"].height,
                        ],
                        "scale_x": scale_x,
                        "scale_y": scale_y,
                    })
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
