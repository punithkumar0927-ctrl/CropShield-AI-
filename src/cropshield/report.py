from __future__ import annotations

import base64
from io import BytesIO
from datetime import datetime, timezone
from html import escape
from typing import Any

from PIL import Image

from .knowledge_base import is_healthy_class
from .treatment_guide import build_action_plan


def build_prediction_report(
    result: dict[str, Any],
    guidance: dict[str, Any] | None,
    processing_seconds: float,
    image: Image.Image | None = None,
) -> str:
    if result.get("status") != "ok":
        raise ValueError("A report can only be generated for a successful prediction.")
    predicted_class = escape(str(result["predicted_class"]))
    confidence = float(result["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Prediction confidence must be between 0 and 1.")
    if processing_seconds < 0:
        raise ValueError("Processing time cannot be negative.")

    top_predictions = "".join(
        "<li>{} — {:.1%}</li>".format(
            escape(str(item["class_name"])),
            float(item["probability"]),
        )
        for item in result.get("top_predictions", [])
    )
    if guidance is None and is_healthy_class(str(result["predicted_class"])):
        guidance_content = (
            "<p>The predicted label is a healthy category. This does not rule out disease or other plant "
            "problems; seek expert advice if symptoms are visible.</p>"
        )
    elif guidance is None:
        guidance_content = (
            "<p>No verified disease-specific guidance is available for this class yet. "
            "Consult a local agricultural officer or qualified plant pathologist before taking action.</p>"
        )
    else:
        guidance_content = (
            f"<h2>How to Treat or Manage This Plant Disease</h2>"
            f"<p>Category: {escape(guidance['disease_category'])}</p>"
            f"<h3>Symptoms</h3><p>{escape(guidance['symptoms'])}</p>"
            f"<h3>Distinguishing signs</h3><p>{escape(guidance['distinguishing_signs'])}</p>"
            f"<h3>Conditions that favor it</h3><p>{escape(guidance['favorable_conditions'])}</p>"
            f"<h3>Possible look-alikes</h3><p>{escape(guidance['look_alikes'])}</p>"
            f"{_list_section('Immediate steps', guidance['immediate_steps'])}"
            f"{_list_section('Management', guidance['management'])}"
            f"{_list_section('What to monitor', guidance['monitoring'])}"
            f"<h2>How to Prevent This Disease</h2>"
            f"{_list_section('Prevention', guidance['prevention'])}"
            f"{_list_section('Safety precautions', guidance['safety_precautions'])}"
            f"<p>Source: <a href=\"{escape(guidance['source_url'], quote=True)}\">"
            f"{escape(guidance['source_name'])}</a>. Last reviewed: "
            f"{escape(guidance['last_reviewed'])}.</p>"
        )

    action_plan = build_action_plan(guidance)
    action_plan_content = "".join(_list_section(title, items) for title, items in action_plan.items())
    image_content = ""
    if image is not None:
        preview = image.convert("RGB")
        preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
        image_buffer = BytesIO()
        preview.save(image_buffer, format="JPEG", quality=75, optimize=True)
        image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        image_content = (
            f'<h2>Submitted image</h2><img alt="Submitted plant image" '
            f'style="max-width:100%;height:auto" src="data:image/jpeg;base64,{image_data}">'
        )
    uncertainty_note = ""
    if result.get("is_uncertain"):
        if (
            result.get("confidence_threshold") is not None
            and confidence < float(result["confidence_threshold"])
        ):
            uncertainty_note = "The result is below the validation-derived confidence threshold."
        else:
            uncertainty_note = "Prediction ambiguity requires extra caution."
    if result.get("scope_warning"):
        uncertainty_note += " This condition may not be covered by the current model; consult a local agricultural expert."
    trust_content = ""
    if result.get("trust_level"):
        trust_content = (
            f"<p><strong>Trust level (heuristic):</strong> {escape(str(result['trust_level']))}<br>"
            f"{escape(str(result.get('trust_explanation', '')))}</p>"
        )
    model_content = (
        f"<br><strong>Model:</strong> {escape(str(result['model_file_name']))}"
        if result.get("model_file_name")
        else ""
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CropShield AI 2.0 prediction report</title>
<style>
body {{ font: 16px/1.5 Arial, sans-serif; color: #17231B; margin: 2rem auto; max-width: 800px; padding: 0 1rem; }}
h1, h2 {{ color: #166534; }} .warning {{ border-left: 4px solid #d97706; padding: .75rem; background: #fff7ed; }}
@media print {{ body {{ margin: 0; }} }}
</style></head>
<body>
<h1>CropShield AI 2.0 — Preliminary Plant Image Report</h1>
<p>Generated {timestamp}</p>
<div class="warning">This is a preliminary model output, not a confirmed diagnosis. PlantVillage-based
models may not generalize to field photographs or unfamiliar conditions. Seek expert advice when uncertain.
{escape(uncertainty_note)}</div>
{image_content}
<h2>Prediction</h2>
<p><strong>Likely class:</strong> {predicted_class}<br>
<strong>Model confidence:</strong> {confidence:.1%}<br>
<strong>Validation threshold:</strong> {f"{float(result['confidence_threshold']):.1%}" if result.get("confidence_threshold") is not None else "Not calibrated"}<br>
<strong>Normalized prediction entropy:</strong> {f"{float(result['normalized_entropy']):.3f}" if result.get("normalized_entropy") is not None else "Not available"}<br>
<strong>Top prediction margin:</strong> {f"{float(result['top_prediction_margin']):.3f}" if result.get("top_prediction_margin") is not None else "Not available"}<br>
<strong>Processing time:</strong> {processing_seconds:.3f} seconds{model_content}</p>
{trust_content}
<h3>Other model outputs</h3><ul>{top_predictions}</ul>
{guidance_content}
<h2>Prioritized action plan</h2>
{action_plan_content}
<p>Do not apply pesticides based solely on this prediction. Follow local guidance and the current product label.</p>
</body></html>"""


def _list_section(title: str, items: list[str]) -> str:
    if not items:
        return ""
    rendered_items = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<h3>{escape(title)}</h3><ul>{rendered_items}</ul>"
