"""
Exports trained YOLOv8n weights to FP32 ONNX and applies Dynamic INT8 Quantization
for Arm Cortex-A72 NEON SIMD acceleration on Raspberry Pi 4 (2GB).
"""

from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_and_quantize(weights_path: str = "runs/detect/train/weights/best.pt"):
    model = YOLO(weights_path)
    fp32_path = model.export(format="onnx", imgsz=320, opset=13, simplify=True)
    int8_path = "../models/neu_yolov8n_int8.onnx"
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"[SUCCESS] FP32 ONNX: {fp32_path} -> Quantized INT8 ONNX: {int8_path}")


if __name__ == "__main__":
    export_and_quantize()
