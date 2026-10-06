"""
bo_loc.py - Các bộ lọc khử nhiễu sử dụng OpenCV.

Mỗi bộ lọc có điểm mạnh/yếu khác nhau tùy loại nhiễu:
- Trung bình: đơn giản, nhanh, làm mờ biên
- Gauss: mượt hơn trung bình, vẫn làm mờ biên
- Trung vị: tốt với nhiễu Muối Tiêu, giữ biên tốt hơn
- Song phương: giữ biên tốt nhất, chậm hơn
"""

import time
import cv2
import numpy as np

# Tên các bộ lọc dùng trong toàn project
TEN_TRUNG_BINH = "Trung bình"
TEN_GAUSS = "Gauss"
TEN_TRUNG_VI = "Trung vị"
TEN_SONG_PHUONG = "Song phương"
DANH_SACH_BO_LOC = [TEN_TRUNG_BINH, TEN_GAUSS, TEN_TRUNG_VI, TEN_SONG_PHUONG]


def _dam_bao_kernel_le(kich_thuoc: int) -> int:
    """OpenCV yêu cầu kích thước kernel là số lẻ và >= 1."""
    kich_thuoc = max(1, int(kich_thuoc))
    if kich_thuoc % 2 == 0:
        kich_thuoc += 1
    return kich_thuoc


def loc_trung_binh(anh: np.ndarray, kich_thuoc_kernel: int = 5) -> tuple[np.ndarray, float]:
    """
    Bộ lọc trung bình (Mean Filter).

    Hoạt động: thay giá trị mỗi pixel bằng trung bình cộng
    của các pixel trong vùng lân cận (kernel).

    cv2.blur() sử dụng box filter - tương đương bộ lọc trung bình.
    """
    k = _dam_bao_kernel_le(kich_thuoc_kernel)
    bat_dau = time.perf_counter()
    ket_qua = cv2.blur(anh, (k, k))
    thoi_gian = time.perf_counter() - bat_dau
    return ket_qua, thoi_gian


def loc_gauss(
    anh: np.ndarray, kich_thuoc_kernel: int = 5, sigma: float = 1.0
) -> tuple[np.ndarray, float]:
    """
    Bộ lọc Gauss (Gaussian Filter).

    Hoạt động: tương tự trung bình nhưng dùng trọng số Gauss.
    Pixel gần tâm có trọng số lớn hơn pixel ở rìa kernel.

    sigma: độ rộng của Gauss (lớn hơn = làm mờ nhiều hơn)
    """
    k = _dam_bao_kernel_le(kich_thuoc_kernel)
    bat_dau = time.perf_counter()
    ket_qua = cv2.GaussianBlur(anh, (k, k), sigmaX=sigma, sigmaY=sigma)
    thoi_gian = time.perf_counter() - bat_dau
    return ket_qua, thoi_gian


def loc_trung_vi(anh: np.ndarray, kich_thuoc_kernel: int = 5) -> tuple[np.ndarray, float]:
    """
    Bộ lọc trung vị (Median Filter).

    Hoạt động: thay giá trị mỗi pixel bằng GIÁ TRỊ TRUNG VỊ
    của các pixel trong vùng lân cận.

    Trung vị không bị ảnh hưởng bởi giá trị ngoại lai (muối/tiêu),
    nên rất hiệu quả với nhiễu Muối Tiêu.
    """
    k = _dam_bao_kernel_le(kich_thuoc_kernel)
    bat_dau = time.perf_counter()
    ket_qua = cv2.medianBlur(anh, k)
    thoi_gian = time.perf_counter() - bat_dau
    return ket_qua, thoi_gian


def loc_song_phuong(
    anh: np.ndarray,
    duong_kinh: int = 9,
    sigma_mau: float = 75.0,
    sigma_khong_gian: float = 75.0,
) -> tuple[np.ndarray, float]:
    """
    Bộ lọc song phương (Bilateral Filter).

    Hoạt động: kết hợp 2 yếu tố:
    - sigma_khong_gian: trọng số theo khoảng cách không gian (như Gauss)
    - sigma_mau: trọng số theo sự khác biệt màu sắc

    Pixel có màu khác biệt lớn (biên) sẽ có trọng số thấp,
    giúp giữ biên trong khi vẫn khử nhiễu vùng đồng nhất.
    """
    duong_kinh = max(1, int(duong_kinh))
    bat_dau = time.perf_counter()
    ket_qua = cv2.bilateralFilter(anh, duong_kinh, sigma_mau, sigma_khong_gian)
    thoi_gian = time.perf_counter() - bat_dau
    return ket_qua, thoi_gian


def ap_dung_tat_ca_bo_loc(
    anh: np.ndarray,
    kich_thuoc_kernel: int = 5,
    sigma_gauss: float = 1.0,
    duong_kinh_song_phuong: int = 9,
    sigma_mau: float = 75.0,
    sigma_khong_gian: float = 75.0,
) -> dict:
    """
    Áp dụng tất cả 4 bộ lọc và trả về kết quả + thời gian xử lý.
    """
    ket_qua = {}

    anh_tb, tg_tb = loc_trung_binh(anh, kich_thuoc_kernel)
    ket_qua[TEN_TRUNG_BINH] = {"anh": anh_tb, "thoi_gian": tg_tb}

    anh_g, tg_g = loc_gauss(anh, kich_thuoc_kernel, sigma_gauss)
    ket_qua[TEN_GAUSS] = {"anh": anh_g, "thoi_gian": tg_g}

    anh_tv, tg_tv = loc_trung_vi(anh, kich_thuoc_kernel)
    ket_qua[TEN_TRUNG_VI] = {"anh": anh_tv, "thoi_gian": tg_tv}

    anh_sp, tg_sp = loc_song_phuong(
        anh, duong_kinh_song_phuong, sigma_mau, sigma_khong_gian
    )
    ket_qua[TEN_SONG_PHUONG] = {"anh": anh_sp, "thoi_gian": tg_sp}

    return ket_qua
