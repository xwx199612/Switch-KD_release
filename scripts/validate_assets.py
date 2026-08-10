from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

EXPECTED_ADAPTER_SHA256 = "e43c568e23f3f19a313f6b1cc65e0d9d9c0bcbc17554d856bce1621657f85e99"
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
EXPECTED_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

def fail(message: str) -> None:
    raise SystemExit(f"HARD CHECK FAILED: {message}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")):
        fail("Hugging Face offline mode is not fully enabled")
    config = Path("/config/runtime.yaml")
    model = Path("/models/student")
    adapter = Path("/models/adapter")
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json", "model.safetensors.index.json"]:
        if not (model / name).is_file(): fail(f"missing Student file: {model / name}")
    if not list(model.glob("*.safetensors")): fail("Student safetensors shards are missing")
    for name in ["deployment_config.json", "adapter_config.json", "adapter_model.safetensors"]:
        if not (adapter / name).is_file(): fail(f"missing adapter file: {adapter / name}")
    deployment = json.loads((adapter / "deployment_config.json").read_text())
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text())
    if deployment.get("artifact_mode") != "4bit_base_bf16_adapter": fail("deployment artifact mode mismatch")
    if deployment.get("base_model_path") != "/models/student": fail("deployment base path is not container-relative")
    if adapter_cfg.get("modules_to_save") != ["model.visual.merger"]: fail("projector modules_to_save mismatch")
    if adapter_cfg.get("base_model_name_or_path") != "/models/student": fail("adapter base path is not container-relative")
    if sha256(adapter / "adapter_model.safetensors") != EXPECTED_ADAPTER_SHA256: fail("adapter SHA256 mismatch")
    if any(shutil.which(name) is None for name in ("cc", "g++", "make")): fail("C/C++ compiler toolchain missing")
    if not config.is_file(): fail("runtime config missing")
    print("asset_hard_checks=PASS")
    print(f"student_model_id={EXPECTED_MODEL_ID}")
    print(f"student_revision={EXPECTED_REVISION}")
    print(f"adapter_sha256={EXPECTED_ADAPTER_SHA256}")
    print("merged_artifact_mode=4bit_base_bf16_adapter")
    print("modules_to_save=model.visual.merger")
    print("offline_mode=PASS")
    print("compiler_toolchain=PASS")

if __name__ == "__main__":
    main()
