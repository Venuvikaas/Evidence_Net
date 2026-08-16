"""Review Package Generator for EVIDENCE-Net (Phase 11).

Generates governed Markdown (report.md) and JSON (report.json) review packages
combining run provenance, tensor artifact metadata, evaluation metrics, and
optional reliability layer presence indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evidence_net.inference.provenance import ProvenanceRecord
from evidence_net.reporting.run_bundle import write_json


def generate_review_package(
    output_dir: Path | str,
    run_id: str,
    *,
    provenance: ProvenanceRecord,
    metrics: dict[str, Any] | None = None,
    artifact_metadata: dict[str, dict[str, Any]] | None = None,
    notes: str = "",
) -> tuple[Path, Path]:
    """Write report.md and report.json into the target output directory.

    Returns (report_md_path, report_json_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_dict = metrics or {}
    art_meta = artifact_metadata or {}
    prov_dict = provenance.as_dict()

    # JSON report payload
    json_payload = {
        "schema": "review-package-v1",
        "run_id": run_id,
        "provenance": prov_dict,
        "metrics": metrics_dict,
        "artifacts": art_meta,
        "notes": notes,
    }

    report_json_path = out_path / "report.json"
    write_json(report_json_path, json_payload)

    # Markdown report formatting
    lines = [
        f"# EVIDENCE-Net Technical Review Package — Run `{run_id}`",
        "",
        "## 1. Provenance Record",
        "",
        "| Component / Contract | Version Identifier |",
        "| --- | --- |",
    ]
    for key, val in prov_dict.items():
        lines.append(f"| `{key}` | `{val}` |")

    lines.extend(
        [
            "",
            "## 2. Artifact Registry",
            "",
            "| Artifact Name | Data Type | Shape | Value Range | SHA-256 Hash |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for art_name, meta in art_meta.items():
        dt = meta.get("dtype", "unknown")
        shp = str(meta.get("shape", []))
        rng = str(meta.get("range", []))
        hsh = meta.get("hash", "not-computed")
        lines.append(f"| `{art_name}` | `{dt}` | `{shp}` | `{rng}` | `{hsh}` |")

    if metrics_dict:
        lines.extend(
            [
                "",
                "## 3. Quantitative Performance Metrics",
                "",
            ]
        )
        for stage, stage_metrics in metrics_dict.items():
            if isinstance(stage_metrics, dict):
                lines.append(f"### Stage: `{stage}`")
                lines.append("")
                lines.append("| Metric | Value | Unit |")
                lines.append("| --- | --- | --- |")
                for m_name, m_val in stage_metrics.items():
                    if isinstance(m_val, dict):
                        v = m_val.get("value", m_val)
                        u = m_val.get("unit", "-")
                        lines.append(f"| `{m_name}` | `{v}` | `{u}` |")
                    else:
                        lines.append(f"| `{m_name}` | `{m_val}` | `-` |")
                lines.append("")

    if notes:
        lines.extend(
            [
                "## 4. Review Notes",
                "",
                notes,
                "",
            ]
        )

    report_md_path = out_path / "report.md"
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report_md_path, report_json_path
