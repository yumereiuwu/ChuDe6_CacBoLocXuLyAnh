"""
thuc_nghiem.py - Chạy thực nghiệm tự động trên nhiều ảnh.

Sử dụng để tạo bảng kết quả thực nghiệm cho báo cáo.
Tất cả chỉ số được tính tự động bởi chương trình.

Chạy: python thuc_nghiem.py
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import pandas as pd

from nhieu import them_nhieu_gauss, them_nhieu_muoi_tieu
from bo_loc import ap_dung_tat_ca_bo_loc, DANH_SACH_BO_LOC
from chi_so import danh_gia_tat_ca

# --- Cấu hình thực nghiệm ---
THU_MUC_HIEN_TAI = os.path.dirname(os.path.abspath(__file__))
THU_MUC_ANH = os.path.join(THU_MUC_HIEN_TAI, "anh")
THU_MUC_KET_QUA = os.path.join(THU_MUC_HIEN_TAI, "ket_qua")
CAU_HINH_NHIEU = {
    "Gauss": {"do_lon": 25},
    "Muối Tiêu": {"ti_le": 0.10},
}
THAM_SO_BO_LOC = {
    "kich_thuoc_kernel": 5,
    "sigma_gauss": 1.0,
    "duong_kinh_song_phuong": 9,
    "sigma_mau": 75,
    "sigma_khong_gian": 75,
}


def chay_thuc_nghiem():
    os.makedirs(THU_MUC_KET_QUA, exist_ok=True)

    danh_sach_anh = [
        f for f in os.listdir(THU_MUC_ANH)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not danh_sach_anh:
        print(f"Không tìm thấy ảnh trong thư mục '{THU_MUC_ANH}/'")
        print("Hãy thêm ít nhất 3 ảnh: chan_dung.jpg, phong_canh.jpg, vat_the.jpg")
        return

    tat_ca_ket_qua = []

    for ten_anh in sorted(danh_sach_anh):
        duong_dan = os.path.join(THU_MUC_ANH, ten_anh)
        anh_goc = cv2.imread(duong_dan)
        if anh_goc is None:
            print(f"  Cảnh báo: không đọc được {duong_dan}")
            continue

        print(f"Đang xử lý: {ten_anh}")

        for ten_nhieu, tham_so in CAU_HINH_NHIEU.items():
            if ten_nhieu == "Gauss":
                anh_nhieu = them_nhieu_gauss(anh_goc, do_lon=tham_so["do_lon"])
            else:
                anh_nhieu = them_nhieu_muoi_tieu(anh_goc, ti_le=tham_so["ti_le"])

            ket_qua_bo_loc = ap_dung_tat_ca_bo_loc(anh_nhieu, **THAM_SO_BO_LOC)
            danh_gia = danh_gia_tat_ca(anh_goc, ket_qua_bo_loc)

            for muc in danh_gia:
                tat_ca_ket_qua.append({
                    "Ảnh": ten_anh,
                    "Loại nhiễu": ten_nhieu,
                    "Bộ lọc": muc["Bộ lọc"],
                    "PSNR (dB)": muc["PSNR (dB)"],
                    "SSIM": muc["SSIM"],
                    "Thời gian (ms)": muc["Thời gian (ms)"],
                })

            thu_muc_luu = os.path.join(
                THU_MUC_KET_QUA,
                ten_anh.replace(".", "_") + f"_{ten_nhieu}",
            )
            os.makedirs(thu_muc_luu, exist_ok=True)
            cv2.imwrite(os.path.join(thu_muc_luu, "goc.jpg"), anh_goc)
            cv2.imwrite(os.path.join(thu_muc_luu, "nhieu.jpg"), anh_nhieu)
            for ten_loc in DANH_SACH_BO_LOC:
                ten_file = ten_loc.lower().replace(" ", "_") + ".jpg"
                cv2.imwrite(
                    os.path.join(thu_muc_luu, ten_file),
                    ket_qua_bo_loc[ten_loc]["anh"],
                )

    if tat_ca_ket_qua:
        bang = pd.DataFrame(tat_ca_ket_qua)
        duong_dan_csv = os.path.join(THU_MUC_KET_QUA, "ket_qua_thuc_nghiem.csv")
        bang.to_csv(duong_dan_csv, index=False, encoding="utf-8-sig")
        print(f"\nHoàn tất! Kết quả lưu tại: {duong_dan_csv}")
        print("\n--- Bảng kết quả ---")
        print(bang.to_string(index=False))
    else:
        print("Không có kết quả.")


if __name__ == "__main__":
    chay_thuc_nghiem()
