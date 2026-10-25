from __future__ import annotations

import base64
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from typing import Any

from PIL import Image


def build_expert_request(
    plant_name: str,
    plot_name: str,
    observations: list[dict[str, Any]],
    selected_observation_id: int,
    observed_symptoms: str,
    trust_level: str,
    location: str | None = None,
    include_selected_photo: bool = False,
    image: Image.Image | None = None,
) -> str:
    selected = next(
        (row for row in observations if int(row["id"]) == selected_observation_id),
        None,
    )
    if selected is None:
        raise ValueError("Select an observation from this plant's history.")
    if not observed_symptoms.strip() or len(observed_symptoms) > 2000:
        raise ValueError("Observed symptoms must contain 1 to 2,000 characters.")
    if trust_level not in {"Low — expert review requested", "Preliminary — not verified"}:
        raise ValueError("Invalid trust level.")
    if location is not None and len(location) > 200:
        raise ValueError("Location must be at most 200 characters.")

    history_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(row["observed_on"])),
            escape(
                "Expert confirmation reported by user — not independently verified"
                if row["record_type"] == "expert_confirmed"
                else str(row["record_type"]).replace("_", " ").title()
            ),
            escape(str(row.get("predicted_class") or "Not recorded")),
            escape(str(row["notes"])),
        )
        for row in observations
    )
    photo_html = "<p>No photo included in this export.</p>"
    if include_selected_photo and image is not None:
        preview = image.convert("RGB")
        preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=78, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        photo_html = (
            '<img alt="User-selected plant photo" style="max-width:100%;height:auto" '
            f'src="data:image/jpeg;base64,{encoded}">'
        )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    selected_prediction = selected.get("predicted_class") or "Not recorded"
    selected_confidence = selected.get("confidence")
    selected_trust = selected.get("trust_level") or "Not recorded"
    selected_record_type = (
        "Expert confirmation reported by user — not independently verified"
        if selected["record_type"] == "expert_confirmed"
        else str(selected["record_type"]).replace("_", " ").title()
    )
    confidence_text = (
        f"{float(selected_confidence):.1%}" if selected_confidence is not None else "Not available"
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CropShield AI expert verification request</title>
<style>
body {{ font: 16px/1.5 Arial,sans-serif; color: #17231B; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
h1,h2 {{ color: #166534; }} .warning {{ padding: .75rem; background: #fff7ed; border-left: 4px solid #d97706; }}
table {{ border-collapse: collapse; width: 100%; }} td,th {{ border: 1px solid #ccd6ca; padding: .5rem; text-align: left; }}
@media print {{ body {{ margin: 0; }} }}
</style></head>
<body>
<h1>Plant-health expert verification request</h1>
<p>Prepared locally: {generated}</p>
<div class="warning">This file is prepared for the user to review and share. CropShield does not send it anywhere.
AI predictions are not confirmed diagnoses. No Grad-CAM heatmap was generated for this export.</div>
<h2>Request details</h2>
<p><strong>Plant:</strong> {escape(plant_name)}<br>
<strong>Plot:</strong> {escape(plot_name)}<br>
<strong>Selected record type:</strong> {escape(selected_record_type)}<br>
<strong>Predicted class:</strong> {escape(str(selected_prediction))}<br>
<strong>Recorded model confidence:</strong> {confidence_text}<br>
<strong>Recorded trust level (heuristic):</strong> {escape(str(selected_trust))}<br>
<strong>Review priority:</strong> {escape(trust_level)}<br>
<strong>Location:</strong> {escape(location or "Not provided")}<br>
<strong>Observed symptoms:</strong> {escape(observed_symptoms)}</p>
<h2>Selected photo</h2>{photo_html}
<h2>Plant history</h2>
<table><thead><tr><th>Date</th><th>Record type</th><th>Model class</th><th>User observation</th></tr></thead>
<tbody>{history_rows}</tbody></table>
<p>Share this report only with an expert you trust. Remove location or other details before sharing if not needed.</p>
</body></html>"""