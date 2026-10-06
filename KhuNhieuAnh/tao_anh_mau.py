"""Tạo ảnh mẫu cho thực nghiệm. Chạy: python tao_anh_mau.py"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

THU_MUC_HIEN_TAI = os.path.dirname(os.path.abspath(__file__))
THU_MUC_ANH = os.path.join(THU_MUC_HIEN_TAI, "anh")
os.makedirs(THU_MUC_ANH, exist_ok=True)

# Ảnh mẫu hình học - mau.jpg (dùng minh họa cơ bản)
duong_dan_mau = os.path.join(THU_MUC_ANH, "mau.jpg")
if not os.path.exists(duong_dan_mau):
    anh = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        anh[i, :, 0] = int(255 * i / 400)
        anh[i, :, 2] = int(255 * (1 - i / 400))
    cv2.circle(anh, (200, 200), 80, (255, 255, 255), -1)
    cv2.rectangle(anh, (50, 50), (150, 150), (0, 255, 255), -1)
    cv2.imwrite(duong_dan_mau, anh)
    print("Đã tạo: mau.jpg")

print(f"Thư mục ảnh mẫu: {THU_MUC_ANH}")
for f in os.listdir(THU_MUC_ANH):
    if f.lower().endswith((".jpg", ".png")):
        print(f" - {f}")
