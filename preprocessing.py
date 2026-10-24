"""Tiền xử lý dùng chung; OpenCV chỉ xử lý ảnh, không học mô hình."""
import cv2
import numpy as np

def preprocess_roi(gray_roi, padding_factor=1.04):
    empty = (None, np.zeros((8, 8), np.uint8), None)
    if gray_roi is None or gray_roi.ndim != 2 or min(gray_roi.shape) < 8:
        return empty
    gray = np.clip(gray_roi, 0, 255).astype(np.uint8)
    if np.percentile(gray, 98) - np.percentile(gray, 2) < 12:
        return empty
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate((blur[0], blur[-1], blur[:, 0], blur[:, -1]))
    # Tự nhận nền trắng/nền tối bằng cường độ ở biên.
    ink = 255 - blur if np.median(border) > 127 else blur
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = mask.shape
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < max(12, h*w*0.0015) or area > h*w*0.60:
            continue
        # Loại vùng chạm mép ROI: thường là bóng, viền giấy hoặc chữ bị cắt.
        if x <= 0 or y <= 0 or x+bw >= w or y+bh >= h:
            continue
        cx, cy = centers[i]
        score = area / (1 + 2*((cx/w-.5)**2 + (cy/h-.5)**2))
        candidates.append((score, i))
    if not candidates:
        return empty
    best = max(candidates)[1]
    x, y, bw, bh, area = stats[best]
    selected = (labels == best).astype(np.uint8)
    # Giữ quầng xám sát nét để resize chống răng cưa; loại nền/nhiễu bên ngoài.
    support = cv2.dilate(selected, np.ones((3,3), np.uint8))
    bg = float(np.median(ink[mask == 0])) if np.any(mask == 0) else 0.
    foreground = np.maximum(ink.astype(np.float32) - bg, 0) * support
    crop = foreground[y:y+bh, x:x+bw]
    side = max(bw, bh)
    size = max(side, int(np.ceil(side*padding_factor)))
    canvas = np.zeros((size, size), np.float32)
    oy, ox = (size-bh)//2, (size-bw)//2
    canvas[oy:oy+bh, ox:ox+bw] = crop
    small = cv2.resize(canvas, (8,8), interpolation=cv2.INTER_AREA)
    peak = float(small.max())
    if peak < 8:
        return empty
    small = np.clip(small/peak, 0, 1)
    return small.reshape(64), (small*255).astype(np.uint8), (x,y,x+bw-1,y+bh-1)

def render_camera_sample(img8, rng):
    """Mô phỏng giấy/ánh sáng từ một ảnh load_digits; không thêm dữ liệu ngoài."""
    up = cv2.resize(img8, (72,72), interpolation=cv2.INTER_CUBIC)
    matrix = cv2.getRotationMatrix2D((35.5,35.5), float(rng.uniform(-10,10)), float(rng.uniform(.82,1.0)))
    up = cv2.warpAffine(up, matrix, (72,72))
    canvas = np.zeros((112,112), np.float32)
    oy, ox = rng.integers(14,26,size=2)
    canvas[oy:oy+72,ox:ox+72] = np.clip(up,0,1)
    paper = float(rng.uniform(185,250)) + np.linspace(-12,12,112)[None,:]
    gray = paper - canvas*float(rng.uniform(100,180))
    gray += rng.normal(0,1.2,gray.shape)
    gray = np.clip(gray,0,255).astype(np.uint8)
    return 255-gray if rng.random()<.25 else gray

