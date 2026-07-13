from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_INITIAL_POLICY_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)


@dataclass(frozen=True)
class CheckpointRegistrationReport:
    is_ready: bool
    checkpoint_path: str
    checkpoint_type: str
    evidence_key: str
    required_files: list[str]
    present_files: list[str]
    missing_files: list[str]
    manifest: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_type": self.checkpoint_type,
            "evidence_key": self.evidence_key,
            "required_files": list(self.required_files),
            "present_files": list(self.present_files),
            "missing_files": list(self.missing_files),
            "manifest": dict(self.manifest),
            "issues": [dict(issue) for issue in self.issues],
        }


def register_initial_policy_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    evidence_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    checkpoint_type: str = "dpo_initial_policy_adapter",
    evidence_key: str = "dpo.initial_policy_checkpoint",
    model_name_or_path: str = "",
    training_config: Mapping[str, Any] | None = None,
) -> CheckpointRegistrationReport:
    base = Path(base_dir) if base_dir is not None else None
    resolved_checkpoint = _resolve_path(checkpoint_path, base_dir=base)
    output_manifest = Path(output_manifest_path)
    required_files = list(REQUIRED_INITIAL_POLICY_FILES)
    present_files: list[str] = []
    missing_files: list[str] = []
    issues: list[dict[str, Any]] = []

    if not resolved_checkpoint.exists():
        issues.append(
            {
                "code": "checkpoint_path_missing",
                "path": str(checkpoint_path),
                "resolved_path": str(resolved_checkpoint),
            }
        )
        missing_files = required_files
    elif not resolved_checkpoint.is_dir():
        issues.append(
            {
                "code": "checkpoint_path_not_directory",
                "path": str(checkpoint_path),
                "resolved_path": str(resolved_checkpoint),
            }
        )
        missing_files = required_files
    else:
        for filename in required_files:
            candidate = resolved_checkpoint / filename
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                present_files.append(filename)
            else:
                missing_files.append(filename)
        for filename in missing_files:
            issues.append(
                {
                    "code": "checkpoint_required_file_missing",
                    "checkpoint_path": str(resolved_checkpoint),
                    "required_file": filename,
                }
            )

    is_ready = not issues
    manifest = {
        "artifact_type": "dpo_initial_policy_checkpoint",
        "checkpoint_type": str(checkpoint_type),
        "checkpoint_path": _portable_path(resolved_checkpoint, base_dir=base),
        "resolved_checkpoint_path": str(resolved_checkpoint),
        "model_name_or_path": str(model_name_or_path),
        "training_config": dict(training_config or {}),
        "required_files": required_files,
        "present_files": present_files,
        "missing_files": missing_files,
        "file_sha256": _file_sha256s(resolved_checkpoint, present_files) if resolved_checkpoint.is_dir() else {},
        "is_ready": is_ready,
    }

    report = CheckpointRegistrationReport(
        is_ready=is_ready,
        checkpoint_path=str(resolved_checkpoint),
        checkpoint_type=str(checkpoint_type),
        evidence_key=str(evidence_key),
        required_files=required_files,
        present_files=present_files,
        missing_files=missing_files,
        manifest=manifest,
        issues=issues,
    )
    _write_json(report.as_dict(), output_manifest)

    if is_ready and evidence_path is not None:
        _upsert_evidence(
            evidence_path=Path(evidence_path),
            evidence_key=str(evidence_key),
            evidence_value=_portable_path(output_manifest, base_dir=base),
        )
    return report


def validate_initial_policy_checkpoint(
    *,
    checkpoint_path: str | Path,
    base_dir: str | Path | None = None,
) -> CheckpointRegistrationReport:
    return register_initial_policy_checkpoint(
        checkpoint_path=checkpoint_path,
        output_manifest_path=Path("/tmp/mias_dcms_checkpoint_validation.json"),
        base_dir=base_dir,
    )


def _resolve_path(path: str | Path, *, base_dir: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        return base_dir / candidate
    return candidate


def _portable_path(path: Path, *, base_dir: Path | None) -> str:
    if base_dir is not None:
        try:
            return path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            return str(path)
    return str(path)


def _file_sha256s(checkpoint_path: Path, filenames: list[str]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for filename in filenames:
        target = checkpoint_path / filename
        if target.is_file():
            digests[filename] = hashlib.sha256(target.read_bytes()).hexdigest()
    return digests


def _upsert_evidence(*, evidence_path: Path, evidence_key: str, evidence_value: str) -> None:
    payload: dict[str, Any] = {}
    if evidence_path.exists():
        with evidence_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("evidence file must contain a JSON object")
        payload.update(loaded)
    payload[evidence_key] = evidence_value
    _write_json(payload, evidence_path)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
