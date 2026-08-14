#!/usr/bin/env python3
"""Run a directory of images through the debug StateTracker for manual review."""

from __future__ import annotations

import argparse
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


def extract_result(response: dict, image_path: Path) -> dict:
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
            "ring_continuity": item.get("ring_continuity"),
            "outer_ring_contrast": item.get("outer_ring_contrast"),
            "background_highlight_evidence": item.get("background_highlight_evidence"),
            "size_ratio": item.get("size_ratio"),
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
        "state_id": response.get("state_id"),
        "is_new": response.get("is_new"),
        "score": response.get("score"),
        "visual_evidence_top": evidence_top,
    }


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
            f"  [{item['index']}] {item['text']} "
            f"score={item['visual_focus_score']:.2f} "
            f"ring_cont={item['ring_continuity']:.2f} "
            f"ring={item['outer_ring_contrast']:.2f} "
            f"bg={item['background_highlight_evidence']:.2f} "
            f"size={item['size_ratio']:.2f}{marker}"
        )
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
                endpoint = "/debug/tracker/start" if not started_tracker else "/debug/tracker/step"
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
                        except (OSError, RequestFailure, subprocess.SubprocessError) as exc:
                            print(
                                f"[WARN] could not save focus image for {image_path.name}: {exc}",
                                file=sys.stderr,
                            )
                    result = extract_result(response, image_path)
                    started_tracker = True
                    results.append(result)
                    print_result(position, len(images), result)
                    if review_writer is not None:
                        write_review_row(review_writer, result)
                        review_handle.flush()
                except (RequestFailure, OSError, KeyError, TypeError, ValueError) as exc:
                    result = {"image": image_path.name, "status": "ERROR_RUNTIME", "error": str(exc)}
                    results.append(result)
                    print(f"[ERROR] {image_path.name}\n{exc}", file=sys.stderr)
                    if review_writer is not None:
                        write_review_row(review_writer, result)
                        review_handle.flush()
                output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
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


if __name__ == "__main__":
    raise SystemExit(main())
