# Khử Nhiễu Ảnh Số — Xử Lý Ảnh Số (Chủ Đề 6)

Ứng dụng web Streamlit nghiên cứu, thực nghiệm và so sánh 4 bộ lọc kinh điển:
**Trung bình (Mean)**, **Gauss (Gaussian)**, **Trung vị (Median)**, **Song phương (Bilateral)**.

## Cấu trúc thư mục

```text
KhuNhieuAnh/
├── ung_dung.py          # Giao diện web tương tác Streamlit chính
├── bo_loc.py            # Cài đặt 4 bộ lọc OpenCV
├── nhieu.py             # Hàm tạo nhiễu Gauss & Muối Tiêu
├── chi_so.py            # Tính toán PSNR, SSIM và thời gian
├── thuc_nghiem.py       # Chạy thực nghiệm tự động xuất CSV
├── tao_anh_mau.py       # Chuẩn bị ảnh mẫu
├── requirements.txt     # Danh sách thư viện Python
├── yeu_cau.txt          # Danh sách thư viện (bản tiếng Việt)
├── HUONG_DAN.md         # Hướng dẫn nhanh
├── anh/                 # Bộ ảnh mẫu benchmark
│   ├── con_cho.jpg      # Ảnh con chó (lông & viền)
│   ├── chan_dung.jpg    # Ảnh chân dung
│   ├── phong_canh.jpg   # Ảnh kiến trúc / phong cảnh
│   ├── vat_the.jpg      # Ảnh vật thể
│   └── mau.jpg          # Ảnh đồ họa gradient
└── ket_qua/             # Kết quả ảnh và file CSV thực nghiệm
    └── ket_qua_thuc_nghiem.csv
```

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run ung_dung.py
```
Mở trình duyệt tại `http://localhost:8501`.

## Chạy thực nghiệm tự động

```bash
python thuc_nghiem.py
```
Kết quả lưu tại `ket_qua/ket_qua_thuc_nghiem.csv`.
