#!/usr/bin/env python3
"""Sequential batch evaluation for the debug StateTracker endpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WHITESPACE_RE = re.compile(r"\s+")
class RequestFailure(Exception):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).strip().lower())


def multipart_image(image_path: Path) -> tuple[bytes, str]:
    boundary = f"----tracker-batch-{uuid.uuid4().hex}"
    data = image_path.read_bytes()
    content_type = "application/octet-stream"
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        content_type = "image/jpeg"
    elif suffix == ".png":
        content_type = "image/png"
    elif suffix == ".webp":
        content_type = "image/webp"
    body = b"--" + boundary.encode() + b"\r\n"
    body += b'Content-Disposition: form-data; name="image"; filename="' + image_path.name.encode() + b'"\r\n'
    body += b"Content-Type: " + content_type.encode() + b"\r\n\r\n"
    body += data + b"\r\n--" + boundary.encode() + b"--\r\n"
    return body, f"multipart/form-data; boundary={boundary}"


def post_json(base_url: str, endpoint: str, image_path: Path | None = None) -> dict:
    url = base_url.rstrip("/") + endpoint
    if image_path is None:
        request = Request(url, data=b"", method="POST")
        request.add_header("Content-Length", "0")
    else:
        body, content_type = multipart_image(image_path)
        request = Request(url, data=body, method="POST")
        request.add_header("Content-Type", content_type)
        request.add_header("Content-Length", str(len(body)))
    try:
        with urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RequestFailure(f"HTTP {exc.code} from {endpoint}", exc.code, body) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RequestFailure(str(exc)) from exc


def runtime_result(sample: dict, failure: RequestFailure) -> dict:
    result = {
        "image": sample.get("image"),
        "gt_focus_text": sample.get("focused_text"),
        "predicted_focus_text": None,
        "focused_index": None,
        "gt_candidate_present": False,
        "status": "ERROR_RUNTIME",
    }
    if failure.status is not None:
        result["http_status"] = failure.status
    if failure.body is not None:
        result["response_body"] = failure.body
    result["exception"] = str(failure)
    return result


def evaluate_response(sample: dict, response: dict) -> dict:
    debug = response.get("tracker_debug") or {}
    parsed = debug.get("parsed_elements") or []
    focused_index = debug.get("focused_index")
    predicted_text = None
    if isinstance(focused_index, int) and 0 <= focused_index < len(parsed):
        if isinstance(parsed[focused_index], dict):
            predicted_text = parsed[focused_index].get("text")

    gt = sample.get("focused_text")
    gt_key = normalize_text(gt) if gt is not None else None
    gt_present = gt_key is not None and any(
        normalize_text(element.get("text")) == gt_key
        for element in parsed
        if isinstance(element, dict)
    )
    if gt is not None and not gt_present:
        status = "FAIL_PARSING_GT_MISSING"
    elif gt is not None and focused_index is None:
        status = "FAIL_FOCUS_FALSE_NEGATIVE"
    elif gt is not None and normalize_text(predicted_text) != gt_key:
        status = "FAIL_FOCUS_WRONG_SELECTION"
    elif gt is None and focused_index is not None:
        status = "FAIL_NO_FOCUS_FALSE_POSITIVE"
    else:
        status = "PASS" if gt is not None else "PASS_NO_FOCUS"

    focus_debug = debug.get("focus_resolver_debug") or {}
    result = {
        "image": sample.get("image"),
        "gt_focus_text": gt,
        "predicted_focus_text": predicted_text,
        "focused_index": focused_index,
        "gt_candidate_present": gt_present,
        "status": status,
        "state_id": response.get("state_id"),
        "is_new": response.get("is_new"),
        "score": response.get("score"),
        "parsing_elapsed_seconds": debug.get("parsing_elapsed_seconds"),
        "focus_elapsed_seconds": debug.get("focus_elapsed_seconds"),
        "total_observation_elapsed_seconds": debug.get("total_observation_elapsed_seconds"),
        "parsed_element_count": len(parsed),
        "focus_image_mode": focus_debug.get("focus_image_mode"),
        "focus_candidate_groups": focus_debug.get("focus_candidate_groups"),
        "focus_candidate_group_types": focus_debug.get("focus_candidate_group_types"),
    }
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def summary_for(results: list[dict]) -> dict:
    counts = {status: sum(result.get("status") == status for result in results) for status in {
        "PASS", "PASS_NO_FOCUS", "FAIL_PARSING_GT_MISSING",
        "FAIL_FOCUS_WRONG_SELECTION", "FAIL_FOCUS_FALSE_NEGATIVE",
        "FAIL_NO_FOCUS_FALSE_POSITIVE",
    }}
    gt_samples = [result for result in results if result.get("gt_focus_text") is not None]
    present_samples = [result for result in gt_samples if result.get("gt_candidate_present")]
    parsing_times = [result["parsing_elapsed_seconds"] for result in results if isinstance(result.get("parsing_elapsed_seconds"), (int, float))]
    focus_times = [result["focus_elapsed_seconds"] for result in results if isinstance(result.get("focus_elapsed_seconds"), (int, float))]
    total_times = [result["total_observation_elapsed_seconds"] for result in results if isinstance(result.get("total_observation_elapsed_seconds"), (int, float))]
    samples = len(results)
    passed = counts["PASS"] + counts["PASS_NO_FOCUS"]
    return {
        "samples": samples,
        "pass": passed,
        "pass_rate": passed / samples if samples else 0.0,
        "parsing_gt_missing": counts["FAIL_PARSING_GT_MISSING"],
        "focus_wrong_selection": counts["FAIL_FOCUS_WRONG_SELECTION"],
        "focus_false_negative": counts["FAIL_FOCUS_FALSE_NEGATIVE"],
        "no_focus_false_positive": counts["FAIL_NO_FOCUS_FALSE_POSITIVE"],
        "parsing_candidate_recall": len(present_samples) / len(gt_samples) if gt_samples else None,
        "conditional_focus_accuracy": sum(result.get("status") == "PASS" for result in present_samples) / len(present_samples) if present_samples else None,
        "avg_parsing_elapsed_seconds": round(mean(parsing_times), 3) if parsing_times else None,
        "avg_focus_elapsed_seconds": round(mean(focus_times), 3) if focus_times else None,
        "avg_total_observation_elapsed_seconds": round(mean(total_times), 3) if total_times else None,
        "p50_focus_elapsed_seconds": percentile(focus_times, 0.50),
        "p95_focus_elapsed_seconds": percentile(focus_times, 0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a JSON array")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with args.output.open("w", encoding="utf-8") as output:
        try:
            post_json(args.base_url, "/debug/tracker/reset")
        except RequestFailure as exc:
            print(f"tracker reset failed: {exc}", file=sys.stderr)

        for position, sample in enumerate(manifest):
            image_name = sample.get("image") if isinstance(sample, dict) else None
            image_path = args.image_dir / image_name if isinstance(image_name, str) else None
            endpoint = "/debug/tracker/start" if position == 0 else "/debug/tracker/step"
            started = time.perf_counter()
            try:
                if image_path is None:
                    raise RequestFailure("manifest sample image must be a string")
                response = post_json(args.base_url, endpoint, image_path)
                result = evaluate_response(sample, response)
            except (RequestFailure, OSError, KeyError, TypeError, ValueError) as exc:
                result = runtime_result(
                    sample,
                    exc if isinstance(exc, RequestFailure) else RequestFailure(str(exc)),
                )
            result.setdefault("batch_elapsed_seconds", round(time.perf_counter() - started, 3))
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()

    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary_for(results), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
