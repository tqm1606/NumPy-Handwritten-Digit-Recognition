"""
Nhận diện chữ số viết tay qua camera.

- Neural network inference viết bằng NumPy, không TensorFlow/Keras/PyTorch.
- Trọng số đi kèm được train từ sklearn.datasets.load_digits bằng train_from_scratch.py.
- Preprocessing được chỉnh để ảnh camera 8x8 gần với load_digits hơn.
"""

from collections import deque
from pathlib import Path
import sys
import argparse
import cv2
import numpy as np

# ============================== CẤU HÌNH ==============================
SOURCE = 0
WEIGHTS_PATH = Path(__file__).with_name("digit_nn_weights.npz")
GUIDE_FRACTION = 0.55

# Dataset load_digits có chữ số gần như chiếm hết chiều cao 8x8.
# Bản cũ dùng 1.35 khiến chữ bị thu nhỏ quá nhiều -> model dễ dồn vào vài lớp.
PADDING_FACTORS = (1.0, 1.04, 1.08)  # test-time ensemble rất nhẹ
SMOOTH_FRAMES = 5                     # trung bình xác suất qua vài frame
MIN_COMPONENT_AREA_RATIO = 0.0015    # bỏ nhiễu nhỏ / khung trống
MIN_DISPLAY_CONFIDENCE = 0.60
# ====================================================================


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class TrainedNetwork:
    def __init__(self, weights_path):
        data = np.load(weights_path)
        self.W = []
        self.b = []
        layer = 1
        while f"W{layer}" in data and f"b{layer}" in data:
            self.W.append(data[f"W{layer}"])
            self.b.append(data[f"b{layer}"])
            layer += 1
        if not self.W:
            raise ValueError("Không tìm thấy W1/b1 trong file trọng số.")

    def predict_proba(self, x64):
        A = np.asarray(x64, dtype=np.float32).reshape(1, -1)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            A = softmax(Z) if i == len(self.W) - 1 else relu(Z)
        return A.ravel()


from preprocessing import preprocess_roi


def predict_with_preprocess_ensemble(net, gray_roi):
    """Trung bình 3 cách crop rất gần nhau để giảm nhạy với bbox 1-2 pixel."""
    probs_list = []
    debug = None
    bbox = None

    for i, factor in enumerate(PADDING_FACTORS):
        x64, dbg, box = preprocess_roi(gray_roi, factor)
        if x64 is None:
            continue
        probs_list.append(net.predict_proba(x64))
        if i == 1 or debug is None:  # ưu tiên factor 1.02 để hiển thị
            debug, bbox = dbg, box

    if not probs_list:
        return None, np.zeros((8, 8), dtype=np.uint8), None

    probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    probs /= np.sum(probs)
    return probs, debug, bbox


def draw_probability_bars(frame, probs, top_left, bar_w=140, bar_h=14, gap=4):
    x0, y0 = top_left
    top = int(np.argmax(probs))
    for d in range(10):
        y = y0 + d * (bar_h + gap)
        cv2.putText(frame, str(d), (x0 - 18, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (70, 70, 70), 1)
        filled = int(bar_w * float(probs[d]))
        color = (60, 90, 230) if d == top else (140, 140, 140)
        cv2.rectangle(frame, (x0, y), (x0 + filled, y + bar_h), color, -1)
        cv2.putText(frame, f"{probs[d]*100:4.1f}%", (x0 + bar_w + 8, y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE))
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    net = TrainedNetwork(WEIGHTS_PATH)

    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Không mở được camera/nguồn video: {SOURCE!r}")
        print("Hãy chạy list_cameras.py rồi sửa SOURCE nếu cần.")
        return

    prob_history = deque(maxlen=SMOOTH_FRAMES)
    print("Đang chạy... Nhấn 'q' để thoát.")

    previous_debug = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Không đọc được khung hình từ camera.")
            break

        if frame.shape[0] < 360 or frame.shape[1] < 480:
            scale = max(360/frame.shape[0], 480/frame.shape[1])
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        fh, fw = frame.shape[:2]
        side = int(min(fh, fw) * GUIDE_FRACTION)
        gx0 = (fw - side) // 2
        gy0 = (fh - side) // 2
        gx1, gy1 = gx0 + side, gy0 + side

        roi_color = frame[gy0:gy1, gx0:gx1]
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        probs, debug8x8, _ = predict_with_preprocess_ensemble(net, roi_gray)

        if probs is None:
            prob_history.clear()
            smooth_probs = None
        else:
            if previous_debug is not None and np.mean(np.abs(debug8x8.astype(float)-previous_debug)) > 35:
                prob_history.clear()
            prob_history.append(probs)
            smooth_probs = np.mean(np.stack(prob_history, axis=0), axis=0)
            smooth_probs /= np.sum(smooth_probs)

        previous_debug = debug8x8.astype(float) if probs is not None else None

        # Khung hướng dẫn
        cv2.rectangle(frame, (gx0, gy0), (gx1, gy1), (255, 255, 255), 2)
        cv2.putText(frame, "dat 1 chu so vao day", (gx0, gy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if smooth_probs is None:
            label = "Chua thay chu so"
            color = (80, 190, 255)
        else:
            pred = int(np.argmax(smooth_probs))
            conf = float(smooth_probs[pred])
            if conf < MIN_DISPLAY_CONFIDENCE or np.sort(smooth_probs)[-1] - np.sort(smooth_probs)[-2] < 0.15:
                label = f"Khong chac: {pred} ({conf*100:.1f}%)"
                color = (80, 190, 255)
            else:
                label = f"{pred}  ({conf*100:.1f}%)"
                color = (60, 90, 230)

        cv2.putText(frame, label, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.35, color, 3, cv2.LINE_AA)

        if smooth_probs is not None:
            draw_probability_bars(frame, smooth_probs, (fw - 200, 90))

        debug_big = cv2.resize(debug8x8, (128, 128), interpolation=cv2.INTER_NEAREST)
        debug_big_bgr = cv2.cvtColor(debug_big, cv2.COLOR_GRAY2BGR)
        frame[10:138, fw - 138:fw - 10] = debug_big_bgr
        cv2.rectangle(frame, (fw - 138, 10), (fw - 10, 138), (255, 255, 255), 1)
        cv2.putText(frame, "model sees 8x8", (fw - 205, 156),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("Nhan dien chu so - nhan q de thoat", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

