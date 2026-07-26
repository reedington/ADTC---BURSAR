"""Fail-fast checks for the ADTC Gate 1 repository contract.

This deliberately does not invent identity, benchmark, model, or media evidence. It turns those
remaining human/hardware tasks into explicit failures instead of allowing a placeholder package
to look complete.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path


PLACEHOLDER_MARKERS = ("TODO", "<<AUTHOR", "your-team", "your-name", "your-email")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}
DOMAINS = {
    "math_scientific_reasoning", "healthcare_medical", "agriculture",
    "creative_writing", "coding_assistants", "corporate_enterprise",
    "autonomous_ai_agents",
}


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str


def _blank(value) -> bool:
    return not isinstance(value, str) or not value.strip()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return False


def _visible_prompts(path: Path) -> list[str]:
    if not path.exists():
        return []
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "visible":
            prompts.append(row.get("prompt", ""))
    return prompts


def check_metadata(root: Path) -> list[Finding]:
    findings = []
    path = root / "metadata.json"
    meta = _read_json(path)
    if meta is None:
        return [Finding("error", "metadata_missing", "metadata.json is missing")]
    if meta is False:
        return [Finding("error", "metadata_json", "metadata.json is not valid JSON")]

    for key in ("team_id", "domain", "language_scope", "african_alpha_claim",
                "budget_laptop_claim", "submitter", "cross_disciplinary_pairing",
                "test_prompts", "model", "_runtime"):
        if key not in meta:
            findings.append(Finding("error", "metadata_field", f"missing metadata field: {key}"))

    if _blank(meta.get("team_id")):
        findings.append(Finding("error", "team_id", "team_id must match the registered ADTF team"))
    if meta.get("domain") not in DOMAINS:
        findings.append(Finding("error", "domain", "domain must be one official identifier"))
    if not isinstance(meta.get("language_scope"), list) or not meta.get("language_scope"):
        findings.append(Finding("error", "language_scope", "language_scope must be a non-empty array"))
    if meta.get("budget_laptop_claim") is not True:
        findings.append(Finding("error", "budget_claim", "budget_laptop_claim must be true"))

    submitter = meta.get("submitter") or {}
    for key in ("name", "email", "github_handle"):
        if _blank(submitter.get(key)):
            findings.append(Finding("error", f"submitter_{key}",
                                    f"submitter.{key} must be filled"))
    if "github" in submitter:
        findings.append(Finding("error", "github_key",
                                "use submitter.github_handle, not submitter.github"))

    pairing = meta.get("cross_disciplinary_pairing") or {}
    if _blank(pairing.get("discipline")) or _blank(pairing.get("description")):
        findings.append(Finding("error", "pairing", "cross-disciplinary pairing is incomplete"))
    if pairing.get("load_bearing") is not True:
        findings.append(Finding("error", "pairing_load", "pairing.load_bearing must be true"))

    prompts = meta.get("test_prompts")
    if not isinstance(prompts, list) or len(prompts) != 2:
        findings.append(Finding("error", "test_prompts", "exactly two test prompts are required"))
        prompt_texts = []
    else:
        prompt_texts = []
        for index, item in enumerate(prompts, 1):
            if _blank(item.get("prompt_id")) or _blank(item.get("prompt")):
                findings.append(Finding("error", "test_prompt_field",
                                        f"test prompt {index} is incomplete"))
            prompt_texts.append(item.get("prompt", ""))
        if len({p.get("prompt_id") for p in prompts}) != 2:
            findings.append(Finding("error", "test_prompt_ids", "test prompt IDs must be unique"))

    visible = _visible_prompts(root / "data/bare/prompts.jsonl")
    if prompt_texts and visible != prompt_texts:
        findings.append(Finding(
            "error", "prompt_sync",
            "metadata test prompts must exactly match the two visible bare-model prompts"))

    model = meta.get("model") or {}
    if model.get("runtime") != "llama.cpp":
        findings.append(Finding("error", "runtime", "model.runtime must be llama.cpp"))
    if not str(model.get("quantization", "")).startswith("GGUF "):
        findings.append(Finding("error", "quantization", "model.quantization must be a GGUF format"))
    if model.get("packaging") not in {"docker_image", "docker_build_from_repo", "binary_bundle"}:
        findings.append(Finding("error", "packaging", "model.packaging is not supported"))

    model_path = (meta.get("_runtime") or {}).get("model_path", "")
    if not model_path.startswith("model/") or not model_path.endswith(".gguf"):
        findings.append(Finding("error", "model_path", "_runtime.model_path must be model/*.gguf"))
    script = root / "download_model.sh"
    if not script.exists():
        findings.append(Finding("error", "download_script", "download_model.sh is missing"))
    elif model_path and Path(model_path).name not in script.read_text(encoding="utf-8"):
        findings.append(Finding(
            "error", "download_path",
            "download_model.sh output filename does not match _runtime.model_path"))

    if meta.get("african_alpha_claim") is True:
        findings.append(Finding(
            "warning", "alpha_evidence",
            "African-language claim requires reviewed, demonstrated model functionality"))
    return findings


def check_artifacts(root: Path, gate1: bool) -> list[Finding]:
    findings = []
    for name in ("README.md", "REPORT.md", "download_model.sh", ".gitignore", "LICENSE"):
        if not (root / name).exists():
            findings.append(Finding("error", "required_file", f"{name} is missing"))

    for name in ("REPORT.md", "data/bare/prompts.jsonl"):
        path = root / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for marker in PLACEHOLDER_MARKERS:
                if marker in text:
                    findings.append(Finding(
                        "error", "placeholder", f"{name} still contains {marker!r}"))

    if gate1:
        if not (root / "submission.json").exists():
            findings.append(Finding(
                "error", "submission_json",
                "submission.json from the target-hardware profiler is missing"))
        media = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
            and ".git" not in p.parts and ".venv" not in p.parts
        ]
        if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in media):
            findings.append(Finding(
                "error", "screenshot", "no application screenshot is present"))
        if not any(p.suffix.lower() in VIDEO_SUFFIXES for p in media):
            findings.append(Finding(
                "error", "video", "the two-minute demonstration video is missing"))
    return findings


def run_checks(root: Path, gate1: bool = False) -> list[Finding]:
    return check_metadata(root) + check_artifacts(root, gate1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--gate1", action="store_true",
                        help="also require profiler output and submission media")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    findings = run_checks(Path(args.root).resolve(), gate1=args.gate1)
    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        for f in findings:
            print(f"{f.level.upper():7} {f.code}: {f.message}")
        errors = sum(f.level == "error" for f in findings)
        warnings = sum(f.level == "warning" for f in findings)
        print(f"\n{errors} error(s), {warnings} warning(s)")
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
