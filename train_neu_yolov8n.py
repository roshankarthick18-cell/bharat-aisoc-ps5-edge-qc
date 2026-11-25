"""
Trains YOLOv8n at 320x320 resolution on the NEU Surface Defect Dataset
Run on Laptop / Google Colab GPU, then export via export_int8_onnx.py
"""

from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="NEU-DET.yaml",
        epochs=80,
        imgsz=320,
        batch=32,
        optimizer="AdamW",
        lr0=0.001,
        project="bharat_aisoc_ps5",
        name="yolov8n_neu_320",
    )


if __name__ == "__main__":
    main()
