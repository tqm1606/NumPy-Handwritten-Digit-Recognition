"""
Train neural network nhận diện chữ số viết tay hoàn toàn FROM SCRATCH bằng NumPy.

Yêu cầu bài tập được tuân thủ:
- Dataset duy nhất: sklearn.datasets.load_digits().
- KHÔNG dùng TensorFlow / Keras / PyTorch.
- KHÔNG dùng model, train_test_split, scaler, metric... của sklearn.
- Forward, backpropagation, Adam, chia train/validation/test và accuracy đều tự cài.

Data augmentation chỉ tạo biến thể từ chính load_digits (xoay/dịch/co giãn nhẹ),
không đưa thêm dataset bên ngoài. Mục đích là giúp model chịu được sai lệch khi
đưa chữ viết tay thật qua camera.
"""

from pathlib import Path
import math
import json
from preprocessing import preprocess_roi, render_camera_sample
import cv2
import numpy as np
from sklearn.datasets import load_digits  # sklearn CHỈ dùng để lấy dataset

SEED = 42
EPOCHS = 55
BATCH_SIZE = 128
AUG_PER_SAMPLE = 7          # 1 ảnh gốc + 7 ảnh biến đổi
LEARNING_RATE_MAX = 2.5e-3
LEARNING_RATE_MIN = 2.0e-4
L2 = 2.0e-4
DROPOUT = 0.10
LABEL_SMOOTHING = 0.02
OUTPUT_WEIGHTS = Path(__file__).with_name("digit_nn_weights.npz")


def stratified_split(y, train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """Tự chia dữ liệu theo từng lớp, không dùng train_test_split của sklearn."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(10):
        idx = np.where(y == cls)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def augment_image(img8, rng):
    """Biến đổi nhẹ ảnh 8x8 từ load_digits để model bền hơn khi dùng camera."""
    up = cv2.resize(img8, (32, 32), interpolation=cv2.INTER_CUBIC)

    angle = float(rng.uniform(-14.0, 14.0))
    scale = float(rng.uniform(0.88, 1.12))
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))

    M = cv2.getRotationMatrix2D((15.5, 15.5), angle, scale)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(
        up, M, (32, 32), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    r = rng.random()
    kernel = np.ones((2, 2), dtype=np.uint8)
    if r < 0.22:
        out = cv2.dilate(out, kernel, iterations=1)   # nét dày hơn
    elif r < 0.38:
        out = cv2.erode(out, kernel, iterations=1)    # nét mảnh hơn

    if rng.random() < 0.40:
        out = cv2.GaussianBlur(out, (3, 3), float(rng.uniform(0.2, 0.8)))

    out = cv2.resize(out, (8, 8), interpolation=cv2.INTER_AREA)
    gamma = float(rng.uniform(0.75, 1.30))
    out = np.clip(out, 0.0, 1.0) ** gamma
    out *= float(rng.uniform(0.85, 1.15))

    if rng.random() < 0.30:
        out += rng.normal(0.0, 0.025, out.shape)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_augmented_set(images, labels, indices, aug_per_sample, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for idx in indices:
        img = images[idx]
        X_list.append(img)
        y_list.append(labels[idx])
        for j in range(aug_per_sample):
            if j % 2 == 0:
                vector, _, _ = preprocess_roi(render_camera_sample(img, rng))
                transformed = vector.reshape(8,8) if vector is not None else augment_image(img, rng)
            else:
                transformed = augment_image(img, rng)
            X_list.append(transformed)
            y_list.append(labels[idx])
    X = np.asarray(X_list, dtype=np.float32).reshape(-1, 64)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class NeuralNetwork:
    """MLP 64 -> 192 -> 96 -> 10, forward/backprop tự cài bằng NumPy."""

    def __init__(self, sizes=(64, 192, 96, 10), seed=7, l2=L2, dropout=DROPOUT):
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.W, self.b = [], []

        # He initialization cho ReLU
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32)
            W *= np.sqrt(2.0 / fan_in)
            self.W.append(W)
            self.b.append(np.zeros((1, fan_out), dtype=np.float32))

        # trạng thái Adam
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X, training=False, rng=None):
        A = X
        activations = [A]
        pre_activations = []
        dropout_masks = []

        for layer in range(len(self.W)):
            Z = A @ self.W[layer] + self.b[layer]
            pre_activations.append(Z)

            if layer == len(self.W) - 1:
                A = softmax(Z)
                dropout_masks.append(None)
            else:
                A = relu(Z)
                mask = None
                if training and self.dropout > 0:
                    mask = (rng.random(A.shape) >= self.dropout).astype(np.float32)
                    mask /= (1.0 - self.dropout)
                    A *= mask
                dropout_masks.append(mask)

            activations.append(A)

        cache = (activations, pre_activations, dropout_masks)
        return A, cache

    def train_batch(self, X, y, lr, rng, label_smoothing=LABEL_SMOOTHING):
        probs, cache = self.forward(X, training=True, rng=rng)
        activations, pre_activations, dropout_masks = cache

        n = len(X)
        num_classes = probs.shape[1]
        target = np.full_like(probs, label_smoothing / (num_classes - 1))
        target[np.arange(n), y] = 1.0 - label_smoothing

        dZ = (probs - target) / n
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)

        for layer in range(len(self.W) - 1, -1, -1):
            A_prev = activations[layer]
            gW[layer] = A_prev.T @ dZ + self.l2 * self.W[layer]
            gb[layer] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 0:
                dA = dZ @ self.W[layer].T
                if dropout_masks[layer - 1] is not None:
                    dA *= dropout_masks[layer - 1]
                dZ = dA * (pre_activations[layer - 1] > 0.0)

        # Adam optimizer tự cài
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gW[i] ** 2)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] ** 2)

            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)

            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def predict_proba(self, X):
        probs, _ = self.forward(X, training=False)
        return probs

    def accuracy(self, X, y):
        pred = np.argmax(self.predict_proba(X), axis=1)
        return float(np.mean(pred == y))

    def snapshot(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def restore(self, snapshot):
        Ws, bs = snapshot
        self.W = [w.copy() for w in Ws]
        self.b = [b.copy() for b in bs]

    def save(self, path):
        np.savez(
            path,
            W1=self.W[0], b1=self.b[0],
            W2=self.W[1], b2=self.b[1],
            W3=self.W[2], b3=self.b[2],
            architecture=np.asarray(self.sizes, dtype=np.int32),
        )


def main():
    print("Đang tải dataset sklearn.datasets.load_digits()...")
    digits = load_digits()
    images = (digits.images.astype(np.float32) / 16.0)
    labels = digits.target.astype(np.int64)

    train_idx, val_idx, test_idx = stratified_split(labels)
    print(f"Train gốc: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")

    print("Đang tạo augmentation chỉ từ dữ liệu load_digits...")
    X_train, y_train = build_augmented_set(
        images, labels, train_idx, AUG_PER_SAMPLE, seed=1234
    )
    X_val = images[val_idx].reshape(-1, 64)
    y_val = labels[val_idx]
    X_test = images[test_idx].reshape(-1, 64)
    y_test = labels[test_idx]

    # validation biến đổi để chọn model không chỉ giỏi ảnh sạch
    X_val_aug, y_val_aug = build_augmented_set(
        images, labels, val_idx, aug_per_sample=2, seed=4321
    )

    print(f"Số mẫu train sau augmentation: {len(X_train)}")
    model = NeuralNetwork()
    rng = np.random.default_rng(999)

    best_score = -1.0
    best_snapshot = None
    best_epoch = 0

    history = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_train))
        progress = (epoch - 1) / max(1, EPOCHS - 1)
        lr = LEARNING_RATE_MIN + (LEARNING_RATE_MAX - LEARNING_RATE_MIN) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            model.train_batch(X_train[idx], y_train[idx], lr, rng)

        clean_val = model.accuracy(X_val, y_val)
        aug_val = model.accuracy(X_val_aug, y_val_aug)
        score = 0.65 * clean_val + 0.35 * aug_val

        history.append({"epoch": epoch, "val_clean": clean_val, "val_augmented": aug_val})
        if score > best_score:
            best_score = score
            best_snapshot = model.snapshot()
            best_epoch = epoch

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | lr={lr:.6f} | "
                f"val sạch={clean_val*100:.2f}% | val biến đổi={aug_val*100:.2f}%"
            )

    model.restore(best_snapshot)
    test_acc = model.accuracy(X_test, y_test)
    aug_val_acc = model.accuracy(X_val_aug, y_val_aug)

    # Test chỉ đánh giá sau khi chọn checkpoint bằng validation.
    X_test_aug, y_test_aug = build_augmented_set(images, labels, test_idx, 2, seed=9876)
    pred = model.predict_proba(X_test).argmax(axis=1)
    confusion = np.zeros((10,10), dtype=int)
    np.add.at(confusion, (y_test, pred), 1)
    metrics = {"seed": SEED, "train_original": len(train_idx), "validation_original": len(val_idx),
               "test_original": len(test_idx), "best_epoch": best_epoch,
               "test_clean_accuracy": test_acc, "test_augmented_accuracy": model.accuracy(X_test_aug,y_test_aug),
               "confusion_matrix": confusion.tolist(), "history": history,
               "note": "Synthetic transformations of load_digits; not a real camera accuracy measurement."}
    OUTPUT_WEIGHTS.with_name("evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save(OUTPUT_WEIGHTS)
    print("\nHoàn tất.")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation biến đổi: {aug_val_acc*100:.2f}%")
    print(f"Test sạch (hold-out): {test_acc*100:.2f}%")
    print(f"Đã lưu trọng số: {OUTPUT_WEIGHTS}")
    print(f"Test gồm ảnh biến đổi: {metrics['test_augmented_accuracy']*100:.2f}%")
    return model, metrics


if __name__ == "__main__":
    main()

