# Bharat AI-SoC Challenge (2026–27) — Problem Statement 05
## Autonomous Closed-Loop Industrial Defect Detection, Upstream Machine Fault Localization & 3-Way Physical Triage System

**Team Members:** N. S. Roshan Karthick, R.Maghizhanban, M. Sabarinathan  
**Institution:** Department of Electronics and Communication Engineering, Velammal Engineering College  
**Target Edge Platform:** Raspberry Pi 4 Model B (2 GB LPDDR4 RAM, Quad-Core Arm Cortex-A72 @ 1.5 GHz) booting via 16 GB USB 3.0 Mass Storage  

### 📦 Included Phase 1 Deliverables in Repository
- **🎥 Live Edge Demo Video:** [`demo/bharat_aisoc_ps5_demo_video.mp4`](demo/bharat_aisoc_ps5_demo_video.mp4)
- **📄 Detailed Technical Report (PDF):** [`docs/Bharat_AI_SoC_PS5_Technical_Report.pdf`](docs/Bharat_AI_SoC_PS5_Technical_Report.pdf)
- **📊 Technical Presentation Deck (PPTX):** [`docs/Bharat_AI_SoC_PS5_Presentation.pptx`](docs/Bharat_AI_SoC_PS5_Presentation.pptx)
- **💻 Complete Edge AI & GPIO Source Code:** [`edge_app/`](edge_app/) & [`train_and_export/`](train_and_export/)

---

## 1. System Overview & Architectural Novelty

This repository implements an air-gapped, real-time Cyber-Physical Quality Control (QC) & Prescriptive Maintenance Assistant engineered for **Problem Statement 05** of the **ARM Bharat AI-SoC Challenge 2026–27**. Rather than stopping at binary defect bounding boxes, our architecture introduces three closed-loop industrial novelties:

1. **Decoupled Dual-Pipeline Asymmetric Core Pinning (Arm Cortex-A72):**
   - **Pipeline 1 (Cores 0–2 — Real-Time Vision):** LAB-space CLAHE hairline crack enhancement + INT8 Quantized **YOLOv8n** (`320x320`) executing on ONNX Runtime / XNNPACK at **18.4 FPS** (`54.3 ms/frame`).
   - **Pipeline 2 (Core 3 — Asynchronous SLM Reasoning):** **Llama 3.2 1B-Instruct** (`8da4w` 4-bit groupwise quantized `.pte`) running via **Meta ExecuTorch** in a non-blocking daemon worker thread with an SQLite LRU response cache (`< 1 ms` cache hit latency).
2. **Spatial-Temporal Centroid Drift & Upstream Machine Origin Localization:**
   - Tracks defect centroids $(c_x, c_y)$ across a 20-frame sliding window. When periodic spatial alignment ($\sigma_x < 12.0\text{ px}$ over $\ge 3$ consecutive parts) or thermal micro-crack bursts occur, the system isolates the exact faulty upstream manufacturing stage (e.g., *Entry Guide Roller #2*, *Quenching Spray Header*, *High-Pressure Descaler Nozzle Bank*).
3. **3-Sector PWM Servo Physical Triage Matrix:**
   - Drives GPIO hardware (IR part-arrival trigger, status LEDs, and an SG90/MG996R PWM diverter gate) to physically route parts into **Pass ($0^\circ$)**, **Sector A: Cosmetic Buffing ($30^\circ$)**, **Sector B: Rework & Grinding ($60^\circ$)**, or **Sector C: Structural Scrap Quarantine ($90^\circ$ + PLC Relay Interlock)**.

---

## 2. Repository Structure

```text
bharat-aisoc-ps5-edge-qc/
├── README.md                          # Complete setup, architecture & benchmark documentation
├── requirements_pi.txt                # Headless dependencies for Raspberry Pi 4 (2GB RAM)
├── setup_pi4_swap.sh                  # Configures 2GB USB 3.0 swap & CPU performance governor
├── train_and_export/
│   ├── train_neu_yolov8n.py           # Trains YOLOv8n (320x320) on NEU Surface Defect dataset
│   ├── export_int8_onnx.py            # Dynamic INT8 quantization for Arm Cortex-A72 NEON
│   └── export_llama_executorch.sh     # Exports Llama 3.2 1B-Instruct (8da4w) to ExecuTorch .pte
└── edge_app/
    ├── gpio_triage_controller.py      # 3-Way PWM Servo Diverter, IR Sensor & LED/Relay driver
    ├── upstream_origin_engine.py      # Spatial-temporal drift tracker & machine origin predictor
    ├── executorch_llama_advisor.py    # Asynchronous ExecuTorch Llama 3.2 1B worker + LRU cache
    ├── logger.py                      # Offline SQLite WAL-mode + CSV audit logger
    ├── dashboard_server.py            # Offline Flask Operator HUD & Machine Tuning Alert UI
    └── edge_inspector.py              # Main multi-threaded real-time inspection orchestrator
```

---

## 3. Upstream Fault Origin & 3-Way Servo Triage Matrix

| NEU Defect Class | Morphological Signature | Predicted Upstream Machine Origin | Severity Tier & Area Criterion | PWM Servo Angle & Physical Triage Sector |
| :--- | :--- | :--- | :--- | :--- |
| **Crazing** | Fine intersecting thermal micro-cracks | Quenching / Cooling Spray Header (Nozzle pressure imbalance) | Minor (`<2.5%`) / Moderate (`2.5-7%`) / Critical (`>=7%`) | `30°` (Sector A) / `60°` (Sector B) / `90°` (Sector C) |
| **Inclusion** | Non-metallic elongated slag pockets | Continuous Casting Tundish / Mold Slag Trap | Moderate (`<5%`) / Critical (`>=5%`) | `60°` (Sector B Rework) / `90°` (Sector C Scrap) |
| **Patches** | Broad localized scale or oil patches | Work Roll Wiper Assembly / Roll Coolant Stripper | Minor (`<3%`) / Moderate (`>=3%`) | `30°` (Sector A Polish) / `60°` (Sector B Grind) |
| **Pitted Surface** | Clustered surface micro-cavities | Pickling Acid Bath / High-Pressure Descaler Header | Moderate (`<6%`) / Critical (`>=6%`) | `60°` (Sector B Rework) / `90°` (Sector C Scrap) |
| **Rolled-in Scale** | Oxide scale pressed into strip | Primary Descaling Pump & Nozzle Bank (< 150 bar) | Moderate (`<5%`) / Critical (`>=5%`) | `60°` (Sector B Descale) / `90°` (Sector C Scrap) |
| **Scratches** | Linear longitudinal abrasion ($\sigma_x < 12\text{ px}$) | Entry/Exit Guide Roller Bearing Seizure or Mechanical Burr | Minor (`<2.5%`) / Moderate (`2.5-7%`) / Critical (`>=7%`) | `30°` (Sector A Buff) / `60°` (Sector B Grind) / `90°` (Sector C Scrap) |

---

## 4. Empirical Edge Benchmarks (Raspberry Pi 4 Model B — 2 GB RAM)

| Model Variant & Precision | Input Resolution | Model Size (MB) | Mean Latency (ms/frame) | Throughput (FPS) | mAP@0.5 (NEU-DET) | Peak Process RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YOLOv8n Baseline (FP32) | 640 x 640 | 12.1 MB | 312.4 ms | 3.20 FPS | 0.768 | 248 MB |
| YOLOv8n Reduced (FP32) | 320 x 320 | 12.1 MB | 118.6 ms | 8.43 FPS | 0.759 | 164 MB |
| **CLAHE + YOLOv8n (INT8 Quantized)** | **320 x 320** | **3.2 MB** | **54.3 ms** | **18.41 FPS** | **0.754** | **92 MB** |
| **Llama 3.2 1B-Instruct (ExecuTorch `8da4w`)** | 128 Context Tokens | 782.0 MB | 5.8 tokens/sec (Async) | Non-blocking | — | **790 MB** |

**Total System Memory Footprint:** `1,066 MB / 2,048 MB` (Leaving **982 MB** free physical RAM headroom).

---

## 5. Quick-Start Deployment on Raspberry Pi 4 (2GB)

```bash
# 1. Configure 2GB swap & Arm Cortex-A72 performance governor
chmod +x setup_pi4_swap.sh
sudo ./setup_pi4_swap.sh

# 2. Install lightweight headless Python dependencies
pip3 install -r requirements_pi.txt

# 3. Run the closed-loop dual-pipeline inspector (with GPIO simulation fallback)
cd edge_app
python3 edge_inspector.py --source 0 --model ../models/neu_yolov8n_int8.onnx --frames 300 --save-snapshots

# 4. (Optional) Launch the local offline operator web HUD on port 5000
python3 dashboard_server.py
```
