# Models

Three pretrained Ultralytics detectors (COCO, 80 classes), exported to ONNX
(640×640 input, opset 12). `backend/app/detection_yolo.py` picks the best one
that the machine can run in reasonable time (`_best_detector`):

| File | Used when | Size | SHA-256 |
|---|---|---|---|
| `yolo11x.onnx` | a DirectX 12 GPU is available (ONNX Runtime DirectML), ~36 ms/frame on an RTX 4060 | 228,057,837 bytes | `de422aad05e4f90dae515c58454cd61b3543037983a7941b8229e83560541eb2` |
| `yolo11l.onnx` | CPU only (OpenCV DNN), ~250 ms/frame | 101,744,083 bytes | `784d41e3b8b119824660b3dc28e7e5f76a3f0d1ae171ffbe1688ac18b792b973` |
| `yolov8n.onnx` | last-resort fallback if the files above are missing | 12,851,144 bytes | `0060a43f6b3371707ab12b2879dd3939267ce508200815ab76b58b8d4b1c8e50` |

None of them needs internet access at run time.

Only these COCO classes are used: `person`, `bicycle`, `car`, `motorcycle`,
`bus`, `truck`.

Why the larger models: on the overhead CCTV clip in `docs/cv-benchmark.md`,
YOLOv8n detected about 17% of the vehicles we counted by hand, against about
86% for YOLO11x. The small model fails mainly on cars partly hidden by trucks,
and on trucks seen from above.

- Source: Ultralytics (`yolo11x.pt`, `yolo11l.pt`, `yolov8n.pt`), exported
  with `YOLO("<name>.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)`.
- Checked after export: the GPU (DirectML) and CPU (OpenCV DNN) paths give the
  same detections on a test frame (`test_gpu_and_cpu_backends_agree`).
- Licence: Ultralytics publishes the YOLO11 and YOLOv8 code and weights under
  **AGPL-3.0** (<https://github.com/ultralytics/ultralytics>). Ultralytics
  also offers a separate commercial licence. We make no claim about what that
  licence requires for any particular use. Read its terms before
  redistributing the models or using them commercially.
- Size: `yolo11x.onnx` is larger than GitHub's 100 MB per-file limit, so a git
  repository needs Git LFS for it. Alternatively, leave it out: the app then
  uses `yolo11l.onnx` (also over 100 MB) or `yolov8n.onnx`.

## Runtime and dependencies

- CPU: OpenCV DNN (`opencv-python-headless`, Apache-2.0, already in
  `requirements.txt`).
- GPU (optional): ONNX Runtime with DirectML (`onnxruntime-directml`, MIT
  licence, Microsoft). It is listed in `requirements-gpu.txt` and installed by
  `start_dashboard.bat` if possible. It runs on any DirectX 12 GPU on Windows
  without a CUDA install. If it's missing or fails its start-up self-test, the
  CPU path is used automatically. Each analysis's technical detail shows the
  active backend and model.

If no model file is present, uploaded clips can only be analysed with the
motion-based detector, and the API reports `object_detector_available: false`
at `/api/health`.
