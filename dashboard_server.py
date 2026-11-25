"""
Offline Local Factory Operator Web HUD & Machine Tuning Alert Dashboard
Reads SQLite WAL inspection telemetry and serves a live status page on port 5000.
"""

import os
import sqlite3
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "../inspection_logs/qc_audit.db")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bharat AI-SoC PS5 — Edge QC & Upstream Fault Origin HUD</title>
  <meta http-equiv="refresh" content="3">
  <style>
    body { font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 24px; }
    .header { background: #1e293b; color: #fff; padding: 16px 24px; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff; }
    th, td { border: 1px solid #cbd5e1; padding: 10px; text-align: left; font-size: 13px; }
    th { background: #f1f5f9; }
    .Critical { color: #dc2626; font-weight: bold; }
    .Moderate { color: #d97706; font-weight: bold; }
    .Minor { color: #2563eb; font-weight: bold; }
  </style>
</head>
<body>
  <div class="header">
    <h2>Bharat AI-SoC PS5: Prescriptive Edge QC & 3-Way Servo Triage HUD</h2>
    <p>Team: N. S. Roshan Karthick, R.Maghizhanban, M. Sabarinathan | Hardware: Raspberry Pi 4B (2GB) Offline Edge</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th>
        <th>Frame</th>
        <th>Defect Class</th>
        <th>Area %</th>
        <th>Severity</th>
        <th>Servo Gate & Sector</th>
        <th>Predicted Faulty Machine Unit</th>
        <th>Operator Machine-Tuning Alert (ExecuTorch Llama 3.2 1B)</th>
      </tr>
    </thead>
    <tbody>
      {% for r in rows %}
      <tr>
        <td>{{ r[0] }}</td>
        <td>{{ r[1] }}</td>
        <td>{{ r[2] }}</td>
        <td>{{ r[3] }}%</td>
        <td class="{{ r[4] }}">{{ r[4] }}</td>
        <td>{{ r[5] }}° ({{ r[6] }})</td>
        <td><strong>{{ r[7] }}</strong></td>
        <td>{{ r[8] }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""


@app.route("/")
def index():
    rows = []
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                """
                SELECT timestamp, frame_id, defect_type, area_pct, severity,
                       servo_angle, triage_sector, predicted_machine_unit, machine_tuning_alert
                FROM inspections ORDER BY id DESC LIMIT 25
                """
            )
            rows = cur.fetchall()
    return render_template_string(HTML_TEMPLATE, rows=rows)


@app.route("/api/latest")
def api_latest():
    if not os.path.exists(DB_PATH):
        return jsonify({"status": "no_logs_yet"})
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    return jsonify({"latest": row})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
