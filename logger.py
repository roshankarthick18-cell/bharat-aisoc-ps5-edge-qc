"""
Offline SQLite (WAL Mode) & CSV Audit Logger for Factory Floor Traceability
"""

import csv
import os
import sqlite3
from datetime import datetime


class InspectionLogger:
    def __init__(self, log_dir: str = "../inspection_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.snapshot_dir = os.path.join(log_dir, "critical_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.db_path = os.path.join(log_dir, "qc_audit.db")
        self.csv_path = os.path.join(log_dir, "qc_audit.csv")
        self._init_storage()

    def _init_storage(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    frame_id INTEGER,
                    defect_type TEXT,
                    confidence REAL,
                    area_pct REAL,
                    spatial_zone TEXT,
                    severity TEXT,
                    servo_angle INTEGER,
                    triage_sector TEXT,
                    predicted_machine_unit TEXT,
                    machine_tuning_alert TEXT,
                    advisory_text TEXT,
                    latency_ms REAL
                )
                """
            )
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_id", "defect_type", "confidence", "area_pct",
                    "spatial_zone", "severity", "servo_angle", "triage_sector",
                    "predicted_machine_unit", "machine_tuning_alert", "latency_ms"
                ])

    def log_event(self, frame_id: int, report: dict, triage: dict, advisory: str, latency_ms: float):
        ts = datetime.now().isoformat(timespec="milliseconds")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO inspections (
                    timestamp, frame_id, defect_type, confidence, area_pct,
                    spatial_zone, severity, servo_angle, triage_sector,
                    predicted_machine_unit, machine_tuning_alert, advisory_text, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, frame_id, report["defect_type"], report["confidence"],
                    report["area_pct"], report["spatial_zone"], report["severity"],
                    triage["servo_angle_deg"], triage["triage_sector"],
                    report["predicted_machine_unit"], report["machine_tuning_alert"],
                    advisory, round(latency_ms, 2)
                ),
            )
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, frame_id, report["defect_type"], report["confidence"],
                report["area_pct"], report["spatial_zone"], report["severity"],
                triage["servo_angle_deg"], triage["triage_sector"],
                report["predicted_machine_unit"], report["machine_tuning_alert"],
                round(latency_ms, 2)
            ])
