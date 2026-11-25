"""
Upstream Machine Origin Prediction & Spatial-Temporal Centroid Drift Engine
Analyzes defect class, bounding-box morphology, and 20-frame coordinate stability
to localize the exact faulty upstream machine unit and prescribe tuning actions.
"""

from collections import deque
import numpy as np

DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

UPSTREAM_MACHINE_MAP = {
    "crazing": {
        "machine_unit": "Unit #4: Quenching & Cooling Spray Header",
        "fault_mechanism": "Thermal shock from uneven cooling nozzle pressure across strip width.",
        "tuning_parameter": "Equalize quenching bank manifold to 4.2 bar; inspect clogged spray nozzles.",
    },
    "inclusion": {
        "machine_unit": "Unit #1: Continuous Casting Tundish & Mold Slag Trap",
        "fault_mechanism": "Non-metallic refractory/slag entrainment pressed into slab matrix.",
        "tuning_parameter": "Skim tundish slag cover; verify submerged entry nozzle (SEN) immersion depth.",
    },
    "patches": {
        "machine_unit": "Unit #3: Finishing Stand Work-Roll Wiper & Stripper",
        "fault_mechanism": "Localized emulsion oil buildup or work-roll surface scaling.",
        "tuning_parameter": "Increase roll wiper pneumatic contact pressure by 10%; flush coolant bar.",
    },
    "pitted_surface": {
        "machine_unit": "Unit #2: Primary Pickling Tank & Descaler Header",
        "fault_mechanism": "Over-pickling chemical pitting or unremoved oxide scale indentation.",
        "tuning_parameter": "Audit acid bath inhibitor concentration and check descaler impingement angle.",
    },
    "rolled_in_scale": {
        "machine_unit": "Unit #2: High-Pressure Hydraulic Descaler Bank",
        "fault_mechanism": "Secondary oxide scale pressed into strip due to header pressure drop.",
        "tuning_parameter": "Boost descaling pump header pressure above 165 bar; purge nozzle bank B.",
    },
    "scratches": {
        "machine_unit": "Unit #5: Entry/Exit Conveyor Guide Roller Assembly",
        "fault_mechanism": "Mechanical abrasion from seized guide roller bearing or sharp guide burr.",
        "tuning_parameter": "Inspect guide roller #2 at matching lateral X-track; deburr and lubricate bearing.",
    },
}


class UpstreamOriginTracker:
    def __init__(self, window_size: int = 20):
        self.history = deque(maxlen=window_size)

    def analyze_defect(self, class_name: str, confidence: float, bbox: tuple, frame_shape: tuple) -> dict:
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        area_pct = ((w * h) / float(max(1, frame_h * frame_w))) * 100.0
        aspect_ratio = round(max(w, h) / float(min(w, h)), 2)

        if cx < frame_w * 0.33:
            spatial_zone = "Left-Track"
        elif cx < frame_w * 0.66:
            spatial_zone = "Center-Track"
        else:
            spatial_zone = "Right-Track"

        # Record in sliding window
        self.history.append({"class": class_name, "cx": cx, "cy": cy, "zone": spatial_zone})

        # Compute consecutive streak and lateral X-coordinate drift (std dev)
        same_class_recent = [item for item in self.history if item["class"] == class_name]
        streak = len(same_class_recent)
        if streak >= 3:
            x_coords = [item["cx"] for item in same_class_recent[-5:]]
            x_drift_std = float(np.std(x_coords))
        else:
            x_drift_std = 99.9

        # Determine severity tier for 3-way physical servo triage
        if area_pct >= 7.0 or (area_pct >= 4.5 and confidence >= 0.78):
            severity = "Critical"
            line_status = "FAIL - SECTOR C (SCRAP)"
        elif area_pct >= 2.5 or confidence >= 0.68:
            severity = "Moderate"
            line_status = "FLAG - SECTOR B (REWORK/GRIND)"
        else:
            severity = "Minor"
            line_status = "WARN - SECTOR A (COSMETIC BUFF)"

        machine_info = UPSTREAM_MACHINE_MAP.get(class_name, {
            "machine_unit": "General Rolling Stand",
            "fault_mechanism": "Unclassified surface anomaly",
            "tuning_parameter": "Perform manual visual calibration.",
        })

        periodic_lock = (streak >= 3 and x_drift_std < 12.0)
        origin_confidence = "HIGH (Periodic Spatial Lock)" if periodic_lock else "MODERATE (Morphological Match)"

        return {
            "defect_type": class_name,
            "confidence": round(confidence, 3),
            "area_pct": round(area_pct, 2),
            "aspect_ratio": aspect_ratio,
            "spatial_zone": spatial_zone,
            "streak_count": streak,
            "x_drift_px": round(x_drift_std, 2),
            "severity": severity,
            "line_status": line_status,
            "predicted_machine_unit": machine_info["machine_unit"],
            "origin_confidence": origin_confidence,
            "fault_mechanism": machine_info["fault_mechanism"],
            "machine_tuning_alert": machine_info["tuning_parameter"],
        }
