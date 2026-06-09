"""Environment compatibility probe for the simulator perception stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import sys
from typing import Any


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    python_version: str
    torch_version: str | None
    cuda_available: bool | None
    ultralytics_import_ok: bool
    ultralytics_error: str | None
    direct_reference_ok: bool
    direct_reference_error: str | None
    hf_download_ok: bool
    hf_download_error: str | None
    checkpoint_load_ok: bool
    checkpoint_error: str | None


def run_checkpoint_compatibility_probe(
    checkpoint_name: str = "openvision/yolo26-n-seg",
    checkpoint_filename: str = "model.pt",
) -> CompatibilityReport:
    torch_version: str | None = None
    cuda_available: bool | None = None
    ultralytics_import_ok = False
    ultralytics_error: str | None = None
    direct_reference_ok = False
    direct_reference_error: str | None = None
    hf_download_ok = False
    hf_download_error: str | None = None
    checkpoint_load_ok = False
    checkpoint_error: str | None = None

    try:
        torch = importlib.import_module("torch")
        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on env state.
        checkpoint_error = f"torch import failed: {exc}"
        return CompatibilityReport(
            python_version=sys.version,
            torch_version=torch_version,
            cuda_available=cuda_available,
            ultralytics_import_ok=ultralytics_import_ok,
            ultralytics_error=ultralytics_error,
            direct_reference_ok=direct_reference_ok,
            direct_reference_error=direct_reference_error,
            hf_download_ok=hf_download_ok,
            hf_download_error=hf_download_error,
            checkpoint_load_ok=checkpoint_load_ok,
            checkpoint_error=checkpoint_error,
        )

    try:
        ultralytics = importlib.import_module("ultralytics")
        ultralytics_import_ok = True
        yolo_cls: Any = getattr(ultralytics, "YOLO")
    except Exception as exc:  # pragma: no cover - depends on env state.
        ultralytics_error = str(exc)
        return CompatibilityReport(
            python_version=sys.version,
            torch_version=torch_version,
            cuda_available=cuda_available,
            ultralytics_import_ok=ultralytics_import_ok,
            ultralytics_error=ultralytics_error,
            direct_reference_ok=direct_reference_ok,
            direct_reference_error=direct_reference_error,
            hf_download_ok=hf_download_ok,
            hf_download_error=hf_download_error,
            checkpoint_load_ok=checkpoint_load_ok,
            checkpoint_error=checkpoint_error,
        )

    try:
        yolo_cls(checkpoint_name)
        direct_reference_ok = True
    except Exception as exc:  # pragma: no cover - depends on env state.
        direct_reference_error = str(exc)

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        hf_hub_download: Any = getattr(huggingface_hub, "hf_hub_download")
        model_path = hf_hub_download(repo_id=checkpoint_name, filename=checkpoint_filename)
        hf_download_ok = True
    except Exception as exc:  # pragma: no cover - depends on env state.
        hf_download_error = str(exc)
        return CompatibilityReport(
            python_version=sys.version,
            torch_version=torch_version,
            cuda_available=cuda_available,
            ultralytics_import_ok=ultralytics_import_ok,
            ultralytics_error=ultralytics_error,
            direct_reference_ok=direct_reference_ok,
            direct_reference_error=direct_reference_error,
            hf_download_ok=hf_download_ok,
            hf_download_error=hf_download_error,
            checkpoint_load_ok=checkpoint_load_ok,
            checkpoint_error=checkpoint_error,
        )

    try:
        yolo_cls(model_path)
        checkpoint_load_ok = True
    except Exception as exc:  # pragma: no cover - depends on env state.
        checkpoint_error = str(exc)

    return CompatibilityReport(
        python_version=sys.version,
        torch_version=torch_version,
        cuda_available=cuda_available,
        ultralytics_import_ok=ultralytics_import_ok,
        ultralytics_error=ultralytics_error,
        direct_reference_ok=direct_reference_ok,
        direct_reference_error=direct_reference_error,
        hf_download_ok=hf_download_ok,
        hf_download_error=hf_download_error,
        checkpoint_load_ok=checkpoint_load_ok,
        checkpoint_error=checkpoint_error,
    )


def main() -> None:
    report = run_checkpoint_compatibility_probe()
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
