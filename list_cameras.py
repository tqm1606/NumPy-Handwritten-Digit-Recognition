"""
Công cụ nhỏ để tìm đúng chỉ số (index) camera của điện thoại trên Windows,
vì máy tính có thể có sẵn webcam tích hợp chiếm index 0.

Chạy:
    python list_cameras.py

Nó sẽ mở lần lượt từng camera tìm thấy (index 0, 1, 2, 3...), hiện preview
trong 3 giây kèm số index ngay trên khung hình. Ghi lại index nào ra hình ảnh
từ điện thoại (qua Iriun/DroidCam) — đó là số bạn cần điền vào SOURCE trong
camera_digit_recognizer.py.

Nhấn 'q' bất cứ lúc nào để bỏ qua camera đang xem và chuyển sang cái tiếp theo.
"""

import cv2
import time
import sys

MAX_INDEX_TO_TRY = 5

for idx in range(MAX_INDEX_TO_TRY):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)  # CAP_DSHOW ổn định hơn trên Windows
    if not cap.isOpened():
        print(f"Index {idx}: không có camera.")
        cap.release()
        continue

    print(f"Index {idx}: tìm thấy camera — đang hiện preview 3 giây (nhấn 'q' để bỏ qua sớm)...")
    start = time.time()
    while time.time() - start < 3:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"SOURCE = {idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Nhan 'q' de qua camera tiep theo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("\nHoàn tất. Dùng đúng số index hiện ra hình ảnh từ điện thoại làm SOURCE trong camera_digit_recognizer.py")

