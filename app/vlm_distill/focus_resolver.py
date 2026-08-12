"""Dedicated single-image navigation-focus resolution."""

from __future__ import annotations

import json
import time
from typing import Any

from PIL import Image, ImageDraw

from .state_tracker import StateObservationError


FOCUS_RESOLVER_PROMPT_TEMPLATE = """Inspect the screenshot and choose the candidate with visible navigation focus.

Candidate numbers are drawn directly on the image. Compare the annotated
candidate containers in their shared visual context; the number in the image
exactly matches the index in the Candidates list.
When shown as numbered tiles, each tile is an enlarged crop of one candidate
from the same screenshot. The original navigation-focus appearance is
preserved; compare border, outline, scale, ring, elevation, and emphasis.

Candidates are already detected UI elements and have been restricted to
geometrically comparable peer groups.
Do not detect new elements.

For every candidate, bbox_norm and size describe the full interactive
container, not just the text. Inspect the annotated candidate containers
directly. Within each visually comparable peer group, compare:

1. full container width and height,
2. scale relative to neighboring peers,
3. visible outer border or outline,
4. focus ring,
5. elevation or container emphasis.

A candidate that is uniquely larger than its neighboring peers is strong focus
evidence. A size difference can be sufficient when it clearly distinguishes one
peer. A visible outer border or outline strengthens that evidence. Also use a
focus ring and elevation or other container emphasis when visible.

Semantic content, recommendation importance, row position, and brightness alone
must not determine focus.

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

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def resolve(self, image: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(candidates, list):
            raise StateObservationError("focus resolver candidates must be a list")

        indices_before_filter = list(range(len(candidates)))
        filtered_indices = self._focus_candidate_indices(candidates)
        filter_used = len(filtered_indices) >= 2
        focus_indices = filtered_indices if filter_used else indices_before_filter
        spatial_order_indices = self._spatial_order_indices(candidates, focus_indices)
        focus_indices = spatial_order_indices
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
            image, focus_candidates, focus_indices, use_montage=filter_used
        )
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
            "focus_candidate_indices_after_filter": filtered_indices
            if filter_used else indices_before_filter,
            "focus_candidate_indices_spatial_order": spatial_order_indices,
            "focus_candidate_filter_used": filter_used,
            "focus_annotation_mode": "index_labels_only",
            "focus_image_mode": focus_image_mode,
            "focus_montage_tile_indices": montage_tile_indices,
            "focus_montage_grid": montage_grid,
            "focus_montage_size": montage_size,
            "focus_montage_tile_sizes": montage_tile_sizes,
        })
        return {
            "focused_index": focused_index,
            "raw_output": raw_output,
            "inference_debug": debug,
        }

    @classmethod
    def _focus_candidate_indices(cls, candidates: list[dict[str, Any]]) -> list[int]:
        """Return candidates with at least one geometrically compatible peer."""
        geometries: dict[int, tuple[float, float, float, float, float, float, float, float]] = {}
        for index, candidate in enumerate(candidates):
            geometry = cls._candidate_geometry(candidate)
            if geometry is not None:
                geometries[index] = geometry

        eligible: list[int] = []
        for index, geometry in geometries.items():
            if any(
                other_index != index
                and cls._geometrically_compatible(geometry, other_geometry)
                for other_index, other_geometry in geometries.items()
            ):
                eligible.append(index)
        return eligible

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
        candidate_boxes: list[tuple[int, int, int, int, int]] = []
        for position, candidate in enumerate(candidates):
            index = candidate_indices[position] if candidate_indices is not None else position
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
                candidate_boxes.append((*pixel_box, index))

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
        for left, top, right, bottom, index in candidate_boxes:
            if right <= roi[0] or left >= roi[2] or bottom <= roi[1] or top >= roi[3]:
                continue
            x1 = max(left, roi[0]) - roi[0]
            y1 = max(top, roi[1]) - roi[1]
            label = str(index)
            text_box = draw.textbbox((0, 0), label)
            label_width = text_box[2] - text_box[0]
            label_height = text_box[3] - text_box[1]
            label_x = min(max(0, x1), max(0, crop.width - label_width - 4))
            if y1 >= label_height + 4:
                label_y = y1 - label_height - 3
            elif x1 >= label_width + 4:
                label_x = x1 - label_width - 3
                label_y = min(max(0, y1), max(0, crop.height - label_height - 4))
            else:
                label_y = min(max(0, y1 + 2), max(0, crop.height - label_height - 4))
            draw.text(
                (label_x, label_y),
                label,
                fill=(255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
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

    @classmethod
    def _prepare_candidate_montage(
        cls,
        image: Image.Image,
        candidate_boxes: list[tuple[int, int, int, int, int]],
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
        """Build enlarged candidate tiles while preserving their original indices."""
        if len(candidate_boxes) < 2:
            return None

        try:
            target_content_height = 300
            montage_max_dimension = 2048
            columns = min(3, len(candidate_boxes))
            gap = 12
            padding = 10
            max_tile_width = (
                montage_max_dimension - (2 * padding) - (columns - 1) * gap
            ) // columns
            tile_data: list[tuple[Image.Image, int, float, int, int]] = []

            for left, top, right, bottom, index in candidate_boxes:
                width = right - left
                height = bottom - top
                margin_x = max(8, round(width * 0.12))
                margin_y = max(8, round(height * 0.12))
                crop_left = max(0, left - margin_x)
                crop_top = max(0, top - margin_y)
                crop_right = min(image.width, right + margin_x)
                crop_bottom = min(image.height, bottom + margin_y)
                crop = image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
                if crop.width <= 0 or crop.height <= 0:
                    return None

                scale = min(
                    target_content_height / crop.height,
                    max_tile_width / crop.width,
                )
                if scale <= 0.0:
                    return None
                resized = crop.resize(
                    (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                tile_data.append((resized, index, scale, crop_left, crop_top))

            tile_width = max(tile.width for tile, *_ in tile_data) + (2 * padding)
            tile_height = max(tile.height for tile, *_ in tile_data) + (2 * padding)
            rows = (len(tile_data) + columns - 1) // columns
            montage_width = columns * tile_width + (columns - 1) * gap
            montage_height = rows * tile_height + (rows - 1) * gap
            if montage_width > montage_max_dimension or montage_height > montage_max_dimension:
                return None

            montage = Image.new("RGB", (montage_width, montage_height), (32, 32, 32))
            annotated_indices: list[int] = []
            tile_sizes: list[list[int]] = []
            for position, (tile, index, scale, crop_left, crop_top) in enumerate(tile_data):
                row, column = divmod(position, columns)
                tile_x = column * (tile_width + gap)
                tile_y = row * (tile_height + gap)
                content_x = tile_x + padding
                content_y = tile_y + padding
                montage.paste(tile, (content_x, content_y))

                candidate_x = content_x + round((candidate_boxes[position][0] - crop_left) * scale)
                candidate_y = content_y + round((candidate_boxes[position][1] - crop_top) * scale)
                draw = ImageDraw.Draw(montage)
                label = str(index)
                text_box = draw.textbbox((0, 0), label)
                label_width = text_box[2] - text_box[0]
                label_height = text_box[3] - text_box[1]
                label_x = min(
                    max(tile_x + 2, candidate_x),
                    tile_x + tile_width - label_width - 3,
                )
                if candidate_y >= label_height + 3:
                    label_y = candidate_y - label_height - 2
                else:
                    label_y = min(
                        max(tile_y + 2, candidate_y + 2),
                        tile_y + tile_height - label_height - 3,
                    )
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=(255, 255, 255),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
                annotated_indices.append(index)
                tile_sizes.append([tile_width, tile_height])

            return (
                montage,
                None,
                False,
                annotated_indices,
                "candidate_montage",
                annotated_indices,
                [rows, columns],
                [montage.width, montage.height],
                tile_sizes,
            )
        except (TypeError, ValueError, OSError):
            return None

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
