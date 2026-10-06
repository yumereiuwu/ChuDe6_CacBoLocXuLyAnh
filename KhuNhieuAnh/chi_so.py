"""
chi_so.py - Tính toán các chỉ số đánh giá chất lượng ảnh.

PSNR (Tỷ số tín hiệu trên nhiễu đỉnh): so sánh cấp pixel
SSIM (Chỉ số tương đồng cấu trúc): so sánh cấu trúc ảnh

Cả hai đều so sánh ảnh đã xử lý với ảnh GỐC SẠCH (không nhiễu).
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tinh_psnr(anh_goc: np.ndarray, anh_xu_ly: np.ndarray) -> float:
    """
    Tính PSNR (Peak Signal-to-Noise Ratio).

    PSNR = 10 * log10(MAX^2 / MSE)
    - MAX = 255 (với ảnh 8-bit)
    - MSE = Sai số bình phương trung bình giữa 2 ảnh

    PSNR cao hơn = ảnh gần với gốc hơn (thường > 30 dB là tốt)
    """
    if anh_goc is None or anh_xu_ly is None:
        return 0.0

    if anh_goc.shape != anh_xu_ly.shape:
        raise ValueError("Hai ảnh phải cùng kích thước để tính PSNR")

    psnr = peak_signal_noise_ratio(anh_goc, anh_xu_ly, data_range=255)
    return round(psnr, 2)


def tinh_ssim(anh_goc: np.ndarray, anh_xu_ly: np.ndarray) -> float:
    """
    Tính SSIM (Structural Similarity Index).

    SSIM so sánh độ sáng, độ tương phản và cấu trúc giữa 2 ảnh.
    Giá trị từ 0 đến 1: 1 = giống hoàn toàn.

    SSIM phản ánh chất lượng cảm nhận tốt hơn PSNR.
    """
    if anh_goc is None or anh_xu_ly is None:
        return 0.0

    if anh_goc.shape != anh_xu_ly.shape:
        raise ValueError("Hai ảnh phải cùng kích thước để tính SSIM")

    if anh_goc.ndim == 3 and anh_goc.shape[2] == 3:
        ssim = structural_similarity(
            anh_goc, anh_xu_ly, channel_axis=2, data_range=255
        )
    else:
        ssim = structural_similarity(anh_goc, anh_xu_ly, data_range=255)

    return round(ssim, 4)


def danh_gia_mot_anh(
    anh_goc: np.ndarray, anh_danh_gia: np.ndarray, ten: str = "Ảnh nhiễu", thoi_gian_s: float = 0.0
) -> dict:
    """Đánh giá chỉ số PSNR, SSIM của một ảnh đơn lẻ so với ảnh gốc."""
    return {
        "Bộ lọc": ten,
        "PSNR (dB)": tinh_psnr(anh_goc, anh_danh_gia),
        "SSIM": tinh_ssim(anh_goc, anh_danh_gia),
        "Thời gian (ms)": round(thoi_gian_s * 1000, 2),
    }


def danh_gia_tat_ca(anh_goc: np.ndarray, ket_qua_bo_loc: dict) -> list[dict]:
    """
    Đánh giá tất cả kết quả bộ lọc so với ảnh gốc.

    Trả về danh sách dict: {Bộ lọc, PSNR, SSIM, Thời gian}
    """
    danh_sach = []

    for ten, du_lieu in ket_qua_bo_loc.items():
        anh_xu_ly = du_lieu["anh"]
        psnr = tinh_psnr(anh_goc, anh_xu_ly)
        ssim = tinh_ssim(anh_goc, anh_xu_ly)
        thoi_gian_ms = round(du_lieu["thoi_gian"] * 1000, 2)

        danh_sach.append({
            "Bộ lọc": ten,
            "PSNR (dB)": psnr,
            "SSIM": ssim,
            "Thời gian (ms)": thoi_gian_ms,
        })

    return danh_sach
