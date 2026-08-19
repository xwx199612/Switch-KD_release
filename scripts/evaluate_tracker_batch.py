#!/usr/bin/env python3
"""Run a directory of images through the debug StateTracker for manual review."""

from __future__ import annotations

import argparse
import os
import csv
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class RequestFailure(Exception):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: str | None = None,
        returncode: int | None = None,
        stderr: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body
        self.returncode = returncode
        self.stderr = stderr


def multipart_image(image_path: Path) -> tuple[bytes, str]:
    boundary = f"----tracker-batch-{uuid.uuid4().hex}"
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "application/octet-stream")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + image_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def post(
    base_url: str,
    endpoint: str,
    image_path: Path | None = None,
    docker_container: str | None = None,
) -> dict:
    if docker_container:
        command = [
            "docker", "exec",
            *( ["-i"] if image_path is not None else [] ),
            docker_container,
            "curl", "-sS", "-X", "POST",
            "--write-out", "\n__TRACKER_HTTP_STATUS__:%{http_code}",
        ]
        if image_path is not None:
            command.extend(["-F", f"image=@-;filename={image_path.name}"])
            input_bytes = image_path.read_bytes()
        else:
            input_bytes = None
        command.append(base_url.rstrip("/") + endpoint)
        try:
            completed = subprocess.run(
                command,
                input=input_bytes,
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RequestFailure(str(exc)) from exc
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        marker = "\n__TRACKER_HTTP_STATUS__:"
        if marker not in stdout:
            detail = f"docker/curl failed with return code {completed.returncode}"
            if stderr:
                detail += f"; stderr: {stderr.strip()}"
            if stdout:
                detail += f"; stdout: {stdout.strip()}"
            raise RequestFailure(
                detail,
                returncode=completed.returncode,
                stderr=stderr,
                body=stdout,
            )
        body, status_text = stdout.rsplit(marker, 1)
        try:
            status = int(status_text.strip())
        except ValueError as exc:
            raise RequestFailure(
                f"invalid HTTP status from docker/curl: {status_text.strip()}",
                returncode=completed.returncode,
                stderr=stderr,
                body=body,
            ) from exc
        if completed.returncode != 0 or not 200 <= status < 300:
            detail = f"HTTP {status} from {endpoint} in container {docker_container}"
            if completed.returncode != 0:
                detail += f"; return code: {completed.returncode}"
            if stderr:
                detail += f"; stderr: {stderr.strip()}"
            if body:
                detail += f"; stdout: {body.strip()}"
            raise RequestFailure(
                detail,
                status=status,
                body=body,
                returncode=completed.returncode,
                stderr=stderr,
            )
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RequestFailure(
                f"invalid JSON from {endpoint} in container {docker_container}: {exc}",
                status=status,
                body=body,
                returncode=completed.returncode,
                stderr=stderr,
            ) from exc

    request = Request(base_url.rstrip("/") + endpoint, method="POST")
    if image_path is None:
        request.data = b""
        request.add_header("Content-Length", "0")
    else:
        body, content_type = multipart_image(image_path)
        request.data = body
        request.add_header("Content-Type", content_type)
        request.add_header("Content-Length", str(len(body)))
    try:
        with urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RequestFailure(f"HTTP {exc.code} from {endpoint}", exc.code, body) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RequestFailure(str(exc)) from exc


def discover_images(image_dir: Path) -> list[Path]:
    return sorted(
        (path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path.relative_to(image_dir)).casefold(),
    )


def _build_element_text_by_index(response: dict[str, Any], fallback_result: dict[str, Any] | None = None) -> dict[int, str]:
    tracker_debug = response.get("tracker_debug") or {}
    elements = tracker_debug.get("parsed_elements") or response.get("parsed_elements")
    if not isinstance(elements, list) and fallback_result:
        elements = fallback_result.get("elements") or fallback_result.get("parsed_elements")
    if not isinstance(elements, list):
        elements = []

    mapping: dict[int, str] = {}
    for position, element in enumerate(elements):
        if isinstance(element, dict):
            raw_index = element.get("index")
            element_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else position
            mapping[element_index] = str(element.get("text") or "")
        else:
            mapping[position] = ""
    return mapping


def _extract_v5_debug_fields(response: dict[str, Any], element_text_by_index: dict[int, str]) -> dict[str, Any]:
    """Extract FocusResolver V5 diagnostics without recomputing decisions."""
    tracker_debug = response.get("tracker_debug") or {}
    focus_debug = tracker_debug.get("focus_resolver_debug") or {}
    v5_keys = {
        "focus_peer_groups", "focus_isolated_indices", "focus_peer_debug",
        "focus_peer_debug_image_path", "focus_cv_prepared_image_path", "focus_cv_prepared_debug_image_path", "focus_cv_prepared_metadata_path", "focus_cv_final_image_path", "focus_enlargement_sibling_groups", "focus_visual_v5_stage",
        "focus_visual_v5_matched", "focus_visual_v5_candidate_index",
        "focus_visual_v5_score", "focus_visual_v5_margin",
        "focus_visual_v5_peer_group_id", "outline_decision",
        "enlargement_decision", "highlight_decision", "isolated_decision",
    }
    available = bool(focus_debug) and any(key in focus_debug for key in v5_keys)

    def get(key: str) -> Any:
        return focus_debug.get(key) if available else None

    index = get("focus_visual_v5_candidate_index")
    text = None
    if isinstance(index, int):
        mapped_text = element_text_by_index.get(index)
        text = mapped_text if mapped_text else f"<unknown #{index}>"
    hierarchy = None if not available else {
        "matched": get("focus_visual_v5_matched"),
        "stage": get("focus_visual_v5_stage"),
        "candidate_index": index,
        "candidate_text": text,
        "score": get("focus_visual_v5_score"),
        "margin": get("focus_visual_v5_margin"),
        "peer_group_id": get("focus_visual_v5_peer_group_id"),
        "reason": get("focus_visual_v5_reason"),
    }
    def normalize_stage(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        best_index = normalized.get("candidate_index", normalized.get("best_index"))
        normalized["best_index"] = best_index
        normalized["best_text"] = (
            element_text_by_index.get(best_index) or f"<unknown #{best_index}>"
            if isinstance(best_index, int) else None
        )
        return normalized

    stages = None if not available else {
        "outline": normalize_stage(get("outline_decision")),
        "enlargement": normalize_stage(get("enlargement_decision")),
        "highlight": normalize_stage(get("highlight_decision")),
        "isolated": normalize_stage(get("isolated_decision")),
    }
    return {
        "v5_debug_available": available,
        "v5_peer_groups": get("focus_peer_groups"),
        "v5_isolated_indices": get("focus_isolated_indices"),
        "v5_peer_debug": get("focus_peer_debug"),
        "v5_peer_debug_image_path": response.get("_peer_debug_host_path") or get("focus_peer_debug_image_path"),
        "cv_prepared_image_path": response.get("_focus_cv_prepared_image_path_host") or get("focus_cv_prepared_image_path"),
        "cv_prepared_debug_image_path": response.get("_focus_cv_prepared_debug_image_path_host") or get("focus_cv_prepared_debug_image_path"),
        "cv_prepared_metadata_path": response.get("_focus_cv_prepared_metadata_path_host") or get("focus_cv_prepared_metadata_path"),
        "cv_final_image_path": response.get("_focus_cv_final_image_path_host") or get("focus_cv_final_image_path"),
        "v5_enlargement_sibling_groups": get("focus_enlargement_sibling_groups"),
        "v5_hierarchy": hierarchy,
        "v5_stages": stages,
        "peer_groups": get("focus_peer_groups"),
        "isolated_indices": get("focus_isolated_indices"),
        "peer_debug_image_path": response.get("_peer_debug_host_path") or get("focus_peer_debug_image_path"),
        "v5_visual_focus_candidate": index,
        "v5_visual_focus_text": text,
        "v5_visual_focus_stage": get("focus_visual_v5_stage"),
        "v5_visual_focus_matched": get("focus_visual_v5_matched"),
        "v5_visual_focus_score": get("focus_visual_v5_score"),
        "v5_visual_focus_margin": get("focus_visual_v5_margin"),
        "v5_visual_focus_peer_group_id": get("focus_visual_v5_peer_group_id"),
        "outline_decision": stages.get("outline") if stages else None,
        "enlargement_decision": stages.get("enlargement") if stages else None,
        "highlight_decision": stages.get("highlight") if stages else None,
        "isolated_decision": stages.get("isolated") if stages else None,
    }


def _extract_result_base(response: dict, image_path: Path) -> dict:
    debug = response.get("tracker_debug") or {}
    elements = debug.get("parsed_elements") or []
    element_texts = [element.get("text", "") for element in elements if isinstance(element, dict)]
    focused_index = debug.get("focused_index")
    focused_text = (
        element_texts[focused_index]
        if isinstance(focused_index, int) and 0 <= focused_index < len(element_texts)
        else None
    )
    focus_debug = debug.get("focus_resolver_debug") or {}
    evidence_by_index = {
        item.get("index"): item
        for item in focus_debug.get("focus_visual_evidence", [])
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    evidence_top = []
    for index in (focus_debug.get("focus_visual_evidence_top_indices") or [])[:5]:
        item = evidence_by_index.get(index)
        if item is None:
            continue
        evidence_top.append({
            "index": index,
            "text": element_texts[index] if 0 <= index < len(element_texts) else "",
            "visual_focus_score": item.get("visual_focus_score"),
            "raw_decoration_score": item.get("raw_decoration_score"),
            "ring_continuity": item.get("ring_continuity"),
            "outer_ring_contrast": item.get("outer_ring_contrast"),
            "background_highlight_evidence": item.get("background_highlight_evidence"),
            "outline_score": item.get("outline_score"),
            "absolute_outline_score": item.get("absolute_outline_score"),
            "raw_outline_score": item.get("raw_outline_score"),
            "outline_exclusivity": item.get("outline_exclusivity"),
            "outline_exclusivity_gate": item.get("outline_exclusivity_gate"),
            "highlight_score": item.get("highlight_score"),
            "enlargement_score": item.get("enlargement_score"),
            "direct_visual_confidence": item.get("direct_visual_confidence"),
            "relative_width": item.get("relative_width"),
            "relative_height": item.get("relative_height"),
            "relative_area": item.get("relative_area"),
            "peer_protrusion_score": item.get("peer_protrusion_score"),
            "uniform_growth": item.get("uniform_growth"),
            "scale_balance": item.get("scale_balance"),
            "size_ratio": item.get("size_ratio"),
            "container_expansion_ratio": item.get("container_expansion_ratio"),
            "focus_scale_signature_version": item.get("focus_scale_signature_version"),
            "scale_signature_score": item.get("scale_signature_score"),
            "scale_width_ratio": item.get("scale_width_ratio"),
            "scale_height_ratio": item.get("scale_height_ratio"),
            "scale_area_ratio": item.get("scale_area_ratio"),
            "scale_isotropy_score": item.get("scale_isotropy_score"),
            "scale_measure_agreement_score": item.get("scale_measure_agreement_score"),
            "scale_peer_reliability": item.get("scale_peer_reliability"),
        })
    return {
        "image": image_path.name,
        "focused_index": focused_index,
        "focused_text": focused_text,
        "elements": element_texts,
        "parsing_elapsed_seconds": debug.get("parsing_elapsed_seconds"),
        "focus_elapsed_seconds": debug.get("focus_elapsed_seconds"),
        "total_observation_elapsed_seconds": debug.get("total_observation_elapsed_seconds"),
        "focus_image_mode": focus_debug.get("focus_image_mode"),
        "focus_visual_evidence_space": focus_debug.get("focus_visual_evidence_space"),
        "state_id": response.get("state_id"),
        "is_new": response.get("is_new"),
        "score": response.get("score"),
        "visual_evidence_top": evidence_top,
    }


def save_peer_debug_image(response: dict[str, Any], image_path: str, output_dir: str, docker_container: str | None = None) -> str | None:
    """Copy the prepared-space V5 peer visualization beside the focus image."""
    tracker_debug = response.get("tracker_debug") or {}
    focus_debug = tracker_debug.get("focus_resolver_debug") or {}
    source = focus_debug.get("focus_peer_debug_image_path")
    if not source:
        return None
    destination = os.path.join(output_dir, f"{Path(image_path).stem}_peers.jpg")
    if docker_container:
        command = ["docker", "exec", docker_container, "cat", source]
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
        with open(destination, "wb") as handle:
            handle.write(completed.stdout)
    else:
        shutil.copyfile(source, destination)
    response["_peer_debug_host_path"] = destination
    debug_sources = (
        ("focus_cv_prepared_image_path", "_cv_prepared.jpg"),
        ("focus_cv_prepared_debug_image_path", "_cv_prepared_debug.jpg"),
        ("focus_cv_prepared_metadata_path", "_cv_prepared.json"),
        ("focus_cv_final_image_path", "_cv_final.jpg"),
    )
    for field, suffix in debug_sources:
        source = focus_debug.get(field)
        if not source:
            continue
        artifact_destination = os.path.join(output_dir, f"{Path(image_path).stem}{suffix}")
        if docker_container:
            completed = subprocess.run(
                ["docker", "exec", docker_container, "cat", source],
                check=True,
                stdout=subprocess.PIPE,
            )
            with open(artifact_destination, "wb") as handle:
                handle.write(completed.stdout)
        else:
            shutil.copyfile(source, artifact_destination)
        response[f"_{field}_host_path"] = artifact_destination
    return destination


def save_focus_image(
    response: dict,
    image_path: Path,
    output_dir: Path,
    docker_container: str | None,
) -> Path:
    focus_debug = (response.get("tracker_debug") or {}).get("focus_resolver_debug") or {}
    debug_path = focus_debug.get("focus_debug_image_path")
    if not isinstance(debug_path, str) or not debug_path:
        raise RequestFailure("FocusResolver did not expose a debug image path")
    if docker_container:
        completed = subprocess.run(
            ["docker", "exec", docker_container, "cat", debug_path],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RequestFailure(
                f"could not read focus debug image from container (return code {completed.returncode})"
                + (f": {stderr}" if stderr else "")
            )
        image_bytes = completed.stdout
    else:
        image_bytes = Path(debug_path).read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{image_path.stem}_focus.png"
    destination.write_bytes(image_bytes)
    return destination


def print_result(position: int, total: int, result: dict) -> None:
    print(f"[{position:03d}/{total:03d}] {result['image']}")
    if result.get("focused_index") is None:
        print("\nFocus:\n  index : null\n  text  : <NO FOCUS>")
    else:
        print(f"\nFocus:\n  index : {result['focused_index']}\n  text  : {result['focused_text']}")
    print("\nVisual evidence:")
    for item in result.get("visual_evidence_top", []):
        marker = "  <-- VLM" if item["index"] == result.get("focused_index") else ""
        print(
            f"  [{item['index']}] {item['text']}\n"
            f"      outline={item['outline_score']:.2f} "
            f"(abs={item['absolute_outline_score']:.2f} "
            f"raw={item['raw_outline_score']:.2f} "
            f"excl={item['outline_exclusivity']:.2f} "
            f"gate={item['outline_exclusivity_gate']:.2f}) "
            f"highlight={item['highlight_score']:.2f} "
            f"enlarge={item['enlargement_score']:.2f} "
            f"rel_w={item['relative_width']:.2f} "
            f"rel_h={item['relative_height']:.2f} "
            f"ug={item['uniform_growth']:.2f} "
            f"balance={item['scale_balance']:.2f} "
            f"protrude={item['peer_protrusion_score']:.2f} "
            f"scale_sig={_format_float(item.get('scale_signature_score'))} "
            f"sw={_format_float(item.get('scale_width_ratio'))} "
            f"sh={_format_float(item.get('scale_height_ratio'))} "
            f"sa={_format_float(item.get('scale_area_ratio'))} "
            f"iso={_format_float(item.get('scale_isotropy_score'))} "
            f"agree={_format_float(item.get('scale_measure_agreement_score'))} "
            f"peer_rel={_format_float(item.get('scale_peer_reliability'))}{marker}"
        )
    print(f"\nEvidence space:\n  {result.get('focus_visual_evidence_space')}")
    print("\nElements:")
    for index, text in enumerate(result.get("elements", [])):
        marker = "  <-- FOCUS" if index == result.get("focused_index") else ""
        print(f"  [{index}] {text}{marker}")
    print(
        "\nTiming:"
        f"\n  parsing : {result.get('parsing_elapsed_seconds')} s"
        f"\n  focus   : {result.get('focus_elapsed_seconds')} s"
        f"\n  total   : {result.get('total_observation_elapsed_seconds')} s"
        f"\n\nMode:\n  {result.get('focus_image_mode')}\n"
    )


def write_review_row(writer: csv.writer, result: dict) -> None:
    writer.writerow([result["image"], result.get("focused_text") or "", "", ""])


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--docker-container")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-from")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--save-focus-images", type=Path)
    parser.add_argument("--debug-dir", type=Path, help="Host directory for prepared CV debug artifacts")
    args = parser.parse_args()

    images = discover_images(args.image_dir)
    if args.start_from:
        start_position = next(
            (position for position, path in enumerate(images)
             if path.name == args.start_from or str(path.relative_to(args.image_dir)) == args.start_from),
            None,
        )
        if start_position is None:
            raise SystemExit(f"--start-from image not found: {args.start_from}")
        images = images[start_position:]
    if args.limit is not None:
        images = images[:max(0, args.limit)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    review_handle = None
    review_writer = None
    if args.review_file:
        args.review_file.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not args.review_file.exists() or args.review_file.stat().st_size == 0
        review_handle = args.review_file.open("a", newline="", encoding="utf-8")
        review_writer = csv.writer(review_handle)
        if needs_header:
            review_writer.writerow(["image", "predicted_focus", "manual_result", "manual_note"])
            review_handle.flush()

    results: list[dict] = []
    started_tracker = False
    try:
        try:
            post(args.base_url, "/debug/tracker/reset", docker_container=args.docker_container)
        except RequestFailure as exc:
            print(f"[ERROR] tracker reset\n{exc}", file=sys.stderr)

        with args.output.open("w", encoding="utf-8") as output:
            for position, image_path in enumerate(images, start=1):
                endpoint = (
                    "/debug/tracker/start"
                    if not started_tracker
                    else "/debug/tracker/step"
                )

                try:
                    response = post(
                        args.base_url,
                        endpoint,
                        image_path,
                        docker_container=args.docker_container,
                    )

                    if args.save_focus_images is not None:
                        try:
                            save_focus_image(
                                response,
                                image_path,
                                args.save_focus_images,
                                args.docker_container,
                            )
                        except (
                            OSError,
                            RequestFailure,
                            subprocess.SubprocessError,
                        ) as exc:
                            print(
                                f"[WARN] could not save focus image for "
                                f"{image_path.name}: {exc}",
                                file=sys.stderr,
                            )

                    if args.debug_dir is not None:
                        args.debug_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        try:
                            save_peer_debug_image(
                                response,
                                str(image_path),
                                str(args.debug_dir),
                                args.docker_container,
                            )
                        except (
                            OSError,
                            RequestFailure,
                            subprocess.SubprocessError,
                        ) as exc:
                            print(
                                f"[WARN] could not save CV debug artifacts for "
                                f"{image_path.name}: {exc}",
                                file=sys.stderr,
                            )

                    result = extract_result(
                        response,
                        image_path,
                    )

                    started_tracker = True
                    results.append(result)
                    print_result(
                        position,
                        len(images),
                        result,
                    )

                    if review_writer is not None:
                        write_review_row(
                            review_writer,
                            result,
                        )
                        review_handle.flush()

                except (
                    RequestFailure,
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    result = {
                        "image": image_path.name,
                        "status": "ERROR_RUNTIME",
                        "error": str(exc),
                    }
                    results.append(result)

                    print(
                        f"[ERROR] {image_path.name}\n{exc}",
                        file=sys.stderr,
                    )

                    if review_writer is not None:
                        write_review_row(
                            review_writer,
                            result,
                        )
                        review_handle.flush()

                output.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                output.flush()
    finally:
        if review_handle is not None:
            review_handle.close()

    parsing = [r["parsing_elapsed_seconds"] for r in results if isinstance(r.get("parsing_elapsed_seconds"), (int, float))]
    focus = [r["focus_elapsed_seconds"] for r in results if isinstance(r.get("focus_elapsed_seconds"), (int, float))]
    total = [r["total_observation_elapsed_seconds"] for r in results if isinstance(r.get("total_observation_elapsed_seconds"), (int, float))]
    print(f"Processed images : {len(results)}")
    print(f"Runtime errors   : {sum(r.get('status') == 'ERROR_RUNTIME' for r in results)}")
    print(f"\nAverage parsing  : {statistics.mean(parsing):.3f} s" if parsing else "\nAverage parsing  : n/a")
    print(f"Average focus    : {statistics.mean(focus):.3f} s" if focus else "Average focus    : n/a")
    print(f"Average total    : {statistics.mean(total):.3f} s" if total else "Average total    : n/a")
    print(f"\nP50 focus        : {percentile(focus, 0.50)} s")
    print(f"P95 focus        : {percentile(focus, 0.95)} s")
    return 0


def _format_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def print_result(*args: object, **kwargs: object) -> None:
    """Compact human-facing V5 presentation; detailed diagnostics stay in JSONL."""
    result = dict(next((value for value in args if isinstance(value, dict)), {}))
    tracker_debug = result.get("tracker_debug") or {}
    debug = tracker_debug.get("focus_resolver_debug") or {}
    if not debug:
        debug = result.get("focus_resolver_debug") or {}
    if isinstance(debug, dict):
        result.setdefault("peer_groups", debug.get("focus_peer_groups", []))
        result.setdefault("isolated_indices", debug.get("focus_isolated_indices", []))
        result.setdefault("peer_debug_image_path", debug.get("focus_peer_debug_image_path"))
        for key in ("outline_decision", "enlargement_decision", "highlight_decision", "isolated_decision"):
            result.setdefault(key, debug.get(key))
        result.setdefault("v5_visual_focus_candidate", debug.get("focus_visual_v5_candidate_index"))
        result.setdefault("v5_visual_focus_stage", debug.get("focus_visual_v5_stage"))
        result.setdefault("v5_visual_focus_matched", debug.get("focus_visual_v5_matched"))
        result.setdefault("v5_visual_focus_score", debug.get("focus_visual_v5_score"))
        result.setdefault("v5_visual_focus_margin", debug.get("focus_visual_v5_margin"))
        result.setdefault("v5_visual_focus_peer_group_id", debug.get("focus_visual_v5_peer_group_id"))
    position = next((value for value in args if isinstance(value, int)), None)
    total = None
    if isinstance(position, int):
        integer_args = [value for value in args if isinstance(value, int)]
        if len(integer_args) > 1:
            total = integer_args[-1]
    image = result.get("image", "<unknown>")
    prefix = f"[{position:03d}/{total:03d}] " if isinstance(position, int) and isinstance(total, int) else ""
    print(f"{prefix}{image}\n")
    print("VLM Decision:")
    print(f"  index : {result.get('focused_index')}")
    print(f"  text  : {result.get('focused_text')}")
    print("\nHierarchy Focus Decision:")
    matched_value = result.get("v5_visual_focus_matched")
    missing_v5 = matched_value is None
    matched = bool(matched_value) if not missing_v5 else False
    print(f"  result : {'UNKNOWN' if missing_v5 else ('MATCH' if matched else 'ABSTAIN')}")
    print(f"  stage  : {result.get('v5_visual_focus_stage') if not missing_v5 else 'UNKNOWN'}")
    index = result.get("v5_visual_focus_candidate") if matched else None
    print(f"  index  : {index}")
    print(f"  text   : {result.get('v5_visual_focus_text') if matched else None}")
    if matched:
        print(f"  score  : {_format_float(result.get('v5_visual_focus_score'))}")
        print(f"  margin : {_format_float(result.get('v5_visual_focus_margin'))}")
        print(f"  peer   : {result.get('v5_visual_focus_peer_group_id')}")
    for name in ("outline", "enlargement", "highlight", "isolated"):
        decision_value = result.get(f"{name}_decision")
        if decision_value is None:
            print(f"  {name:<11}: UNKNOWN")
            continue
        decision = decision_value or {}
        if not decision.get("executed", False):
            print(f"  {name:<11}: SKIPPED")
        else:
            status = "MATCH" if decision.get("matched") else ("ABSTAIN" if name == "isolated" else "NO_HIT")
            print(f"  {name:<11}: {status:<7} best=#{decision.get('candidate_index')} score={_format_float(decision.get('score'))} margin={_format_float(decision.get('margin'))}")
    print("\nPeer debug:")
    print(f"  groups   : {len(result.get('peer_groups') or [])}")
    print(f"  isolated : {len(result.get('isolated_indices') or [])}")
    print(f"  image    : {result.get('peer_debug_image_path') or '-'}")
    print("\nTiming: parsing={}s focus={}s total={}s".format(_format_float(result.get("parsing_elapsed_seconds")), _format_float(result.get("focus_elapsed_seconds")), _format_float(result.get("total_observation_elapsed_seconds"))))
    print(f"Mode: {result.get('focus_image_mode') or '-'}")


def extract_result(response: dict, image_path: Path) -> dict:
    result = _extract_result_base(response, image_path)
    element_text_by_index = _build_element_text_by_index(response, result)
    result["element_text_by_index"] = element_text_by_index
    raw_elements = result.get("elements") or result.get("parsed_elements") or []
    if isinstance(raw_elements, list):
        result["element_texts"] = [
            str(element.get("text") or "") if isinstance(element, dict) else ""
            for element in raw_elements
        ]
    else:
        result["element_texts"] = [element_text_by_index[index] for index in sorted(element_text_by_index)]
    focused_index = result.get("focused_index")
    if isinstance(focused_index, int):
        result["focused_text"] = element_text_by_index.get(focused_index) or f"<unknown #{focused_index}>"
    result.update(_extract_v5_debug_fields(response, element_text_by_index))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
