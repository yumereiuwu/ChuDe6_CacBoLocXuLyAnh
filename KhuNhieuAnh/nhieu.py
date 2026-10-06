"""
nhieu.py - Tạo nhiễu cho ảnh sử dụng NumPy.

Hai loại nhiễu phổ biến trong xử lý ảnh:
1. Nhiễu Gauss: nhiễu ngẫu nhiên theo phân phối chuẩn
2. Nhiễu Muối Tiêu: các pixel trắng (muối) và đen (tiêu) ngẫu nhiên
"""

import numpy as np


def them_nhieu_gauss(anh: np.ndarray, do_lon: float) -> np.ndarray:
    """
    Thêm nhiễu Gauss vào ảnh.

    Công thức: anh_nhieu = anh + N(0, do_lon^2)
    - do_lon (sigma): độ lớn của nhiễu (1-50, thường dùng 10-30)

    Nhiễu Gauss mô phỏng nhiễu từ cảm biến camera, điều kiện ánh sáng yếu.
    """
    if anh is None or anh.size == 0:
        raise ValueError("Ảnh không hợp lệ")

    nhieu = np.random.normal(0, do_lon, anh.shape)
    anh_nhieu = anh.astype(np.float64) + nhieu
    anh_nhieu = np.clip(anh_nhieu, 0, 255)

    return anh_nhieu.astype(np.uint8)


def them_nhieu_muoi_tieu(anh: np.ndarray, ti_le: float) -> np.ndarray:
    """
    Thêm nhiễu Muối Tiêu vào ảnh.

    - ti_le: tỷ lệ pixel bị nhiễu (0.0 - 0.5)
    - Một nửa pixel nhiễu là muối (255), nửa còn lại là tiêu (0)

    Nhiễu Muối Tiêu thường xuất hiện do lỗi truyền dữ liệu hoặc cảm biến bị lỗi.
    """
    if anh is None or anh.size == 0:
        raise ValueError("Ảnh không hợp lệ")

    ti_le = np.clip(ti_le, 0.0, 0.5)
    anh_nhieu = anh.copy()

    # Sinh ma trận phân bố đều ngẫu nhiên trong khoảng [0, 1)
    ma_tran_ngau_nhien = np.random.random(anh.shape[:2])
    # Một nửa tỷ lệ nhiễu là muối (trắng = 255)
    mat_na_muoi = ma_tran_ngau_nhien < (ti_le / 2.0)
    # Nửa tỷ lệ nhiễu còn lại là tiêu (đen = 0)
    mat_na_tieu = (ma_tran_ngau_nhien >= (ti_le / 2.0)) & (ma_tran_ngau_nhien < ti_le)

    anh_nhieu[mat_na_muoi] = 255
    anh_nhieu[mat_na_tieu] = 0

    return anh_nhieu
