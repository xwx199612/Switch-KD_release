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
    VISUAL_RING_CONTINUITY_WEIGHT = 0.55
    VISUAL_RING_CONTRAST_WEIGHT = 0.30
    VISUAL_BACKGROUND_WEIGHT = 0.15
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
        visual_evidence = self._visual_focus_evidence(image, candidates, candidate_groups)
        # Grouping is descriptive and controls presentation order; it must not
        # remove candidates from the resolver input.
        filter_used = False
        filtered_indices = list(focus_indices)
        spatial_order_indices = list(focus_indices)
        focus_candidates = [candidates[index] for index in focus_indices]

        (
            annotated_image,
            roi_bbox,
            roi_used,
            annotated_indices,
            focus_image_mode,
            montage_tile_indices,
            montage_grid,
            montage_size,
            montage_tile_sizes,
        ) = self._prepare_focus_image(
            image,
            focus_candidates,
            focus_indices,
            focus_group_ids=focus_group_ids,
            use_montage=len(candidate_groups) >= 2,
        )
        focus_debug_image_path: str | None = None
        try:
            annotated_image.save(self.DEBUG_IMAGE_PATH, format="PNG")
            focus_debug_image_path = self.DEBUG_IMAGE_PATH
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
            "focus_visual_evidence": visual_evidence,
            "focus_visual_evidence_top_indices": [
                item["index"] for item in sorted(
                    visual_evidence,
                    key=lambda item: (-item["visual_focus_score"], item["index"]),
                )
            ],
        })
        return {
            "focused_index": focused_index,
            "raw_output": raw_output,
            "inference_debug": debug,
        }

    @classmethod
    def _visual_focus_evidence(
        cls,
        image: Image.Image,
        candidates: list[dict[str, Any]],
        candidate_groups: list[list[int]],
    ) -> list[dict[str, Any]]:
        """Compute diagnostic-only focus decoration evidence from the source image."""
        geometries = {
            index: cls._candidate_pixel_geometry(image, candidate)
            for index, candidate in enumerate(candidates)
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
                ring_continuity, ring_contrast, background_delta, edge_strength = (
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
                "edge": best["edge"],
                "area": float(width * height),
                "width": float(width),
                "height": float(height),
                "group_id": float(group_id),
            }

        evidence: list[dict[str, Any]] = []
        for index in range(len(candidates)):
            values = raw[index]
            peer_indices = [
                peer for peer in candidate_groups[group_by_index[index]]
                if peer in raw
            ] if index in group_by_index else [index]
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
            score = (
                cls.VISUAL_RING_CONTINUITY_WEIGHT * continuity
                + cls.VISUAL_RING_CONTRAST_WEIGHT * ring
                + cls.VISUAL_BACKGROUND_WEIGHT * background
            )
            evidence.append({
                "index": index,
                "group_id": group_by_index.get(index, -1),
                "visual_cell_bbox": values["visual_cell"],
                "selected_container_bbox": values["selected_container"],
                "container_expansion_ratio": round(values["expansion_ratio"], 4),
                "raw_decoration_score": round(values["raw_score"], 4),
                "ring_continuity": round(continuity, 4),
                "outer_ring_contrast": round(ring, 4),
                "background_highlight_evidence": round(background, 4),
                "size_ratio": round(size_ratio, 4),
                "width_ratio": round(width_ratio, 4),
                "height_ratio": round(height_ratio, 4),
                "local_edge_strength": round(edge, 4),
                "visual_focus_score": round(score, 4),
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
    ) -> tuple[float, float, float, float]:
        ring_continuity, ring_contrast = cls._perimeter_ring_evidence(image, box)
        background_mean = cls._mean_luma_band(image, box, 14, 5)
        perimeter_mean = cls._perimeter_luma(image, box)
        background_delta = cls._clamp01(
            abs(perimeter_mean - background_mean) / 255.0
        )
        edge_strength = cls._perimeter_edge_strength(image, box)
        return ring_continuity, ring_contrast, background_delta, edge_strength

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
            if top >= 7:
                add_sample(0, position, (x, top + 1), (x, top - 2), (x, top - 7), True)
            if bottom + 7 < image.height:
                add_sample(1, position, (x, bottom - 1), (x, bottom + 2), (x, bottom + 7), True)
            if left >= 7:
                add_sample(2, position, (left + 1, y), (left - 2, y), (left - 7, y), False)
            if right + 7 < image.width:
                add_sample(3, position, (right - 1, y), (right + 2, y), (right + 7, y), False)

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
        tuple[int, int, int, int] | None,
        bool,
        list[int],
        str,
        list[int],
        list[int],
        list[int],
        list[list[int]],
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
        draw = ImageDraw.Draw(crop)
        annotated_indices: list[int] = []
        for left, top, right, bottom, index, _group_id in candidate_boxes:
            if right <= roi[0] or left >= roi[2] or bottom <= roi[1] or top >= roi[3]:
                continue
            x1 = max(left, roi[0]) - roi[0]
            y1 = max(top, roi[1]) - roi[1]
            cls._draw_focus_index_label(draw, index, x1, y1, crop.size)
            annotated_indices.append(index)

        input_image = crop
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
        return (
            input_image,
            roi,
            roi_used,
            annotated_indices,
            "roi" if roi_used else "full_image",
            [],
            [0, 0],
            [0, 0],
            [],
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
        tuple[int, int, int, int] | None,
        bool,
        list[int],
        str,
        list[int],
        list[int],
        list[int],
        list[list[int]],
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
            annotated_indices: list[int] = []
            tile_sizes: list[list[int]] = []
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
                    montage.paste(tile, (x, y))
                    tile_sizes.append([tile_width, tile_height])
                    x += tile_width + gap
                y += row_heights[row] + gap

            return (
                montage,
                None,
                False,
                annotated_indices,
                "group_montage",
                annotated_indices,
                [rows, columns],
                [montage.width, montage.height],
                tile_sizes,
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
