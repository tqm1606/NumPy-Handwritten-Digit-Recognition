# Handwritten Digit Recognition — NumPy from Scratch

A handwritten digit classifier and camera demo built without TensorFlow, Keras, PyTorch, pretrained weights, or a scikit-learn model. Scikit-learn is used only to load the `load_digits` dataset.

## Architecture

- Dataset: `sklearn.datasets.load_digits()` — 1,797 images, 8×8 pixels, classes 0–9.
- MLP: 64 → 192 → 96 → 10.
- NumPy implementation of forward propagation, backpropagation, Adam, stratified splitting, and accuracy.
- OpenCV preprocessing and camera interface.
- Split: 1,252 train, 266 validation, 279 clean hold-out test; seed 42.

## Results

| Evaluation | Result |
|---|---:|
| Clean hold-out test | **276/279 (98.92%)** |
| Synthetic transformed test | **97.49%** |
| Simulated camera ROI | **277/279 (99.28%)** |

The 98.92% clean-test result was recomputed from the included weights and deterministic split. The 99.28% result is labeled simulated ROI so it remains distinct from a physical-webcam benchmark.

`evaluation.json` contains validation history, the clean-test confusion matrix, and transformed-test result. `comparison.json` contains the simulated-ROI comparison. See `results.csv` for a compact summary.

## Run

```bash
python -m pip install -r requirements.txt
python verify.py
python train_from_scratch.py
```

Camera demo:

```bash
python list_cameras.py
python camera_digit_recognizer.py --source 0
```

## Tóm tắt

Mạng MLP 64–192–96–10 được cài đặt bằng NumPy. Trọng số đi kèm đạt **276/279 = 98,92%** trên tập test sạch giữ riêng. Kết quả **99,28%** thuộc bài kiểm tra ROI mô phỏng và được ghi nhãn riêng.
