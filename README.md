# BÀI GIỮA KỲ MÔN XỬ LÝ ẢNH SỐ
## CHỦ ĐỀ 6: CÁC BỘ LỌC XỬ LÝ ẢNH & SO SÁNH KHẢ NĂNG KHỬ NHIỄU

---

## 1. Tên Đề Tài & Mục Tiêu

### 1.1. Tên đề tài
**Nghiên cứu, cài đặt và so sánh thực nghiệm 4 bộ lọc xử lý ảnh kinh điển: Mean Filter, Gaussian Filter, Median Filter, Bilateral Filter.**

### 1.2. Mục tiêu nghiên cứu
- Tìm hiểu bản chất toán học và nguyên lý hoạt động của 4 bộ lọc xử lý ảnh trong miền không gian.
- Xây dựng mô hình tạo nhiễu ảnh thực tế: **Nhiễu Gauss (Gaussian Noise)** và **Nhiễu Muối Tiêu (Salt & Pepper Noise)**.
- Thiết kế quy trình thực nghiệm chuẩn:
  $$\text{Ảnh gốc sạch} \longrightarrow \text{Tạo nhiễu} \longrightarrow \text{Ảnh bị nhiễu} \longrightarrow \text{4 Bộ lọc} \longrightarrow \text{So sánh & Đánh giá}$$
- Đánh giá định lượng bằng các chỉ số khoa học: **PSNR (Peak Signal-to-Noise Ratio)**, **SSIM (Structural Similarity Index)** và **Thời gian xử lý (Execution Time - ms)** so với ảnh gốc sạch đối chứng.
- Phân tích khả năng **bảo toàn đường biên cạnh (Edge Preservation)** bằng công cụ soi phóng to vùng chi tiết (ROI Zoom).
- Cung cấp giao diện trực quan hóa bằng **Streamlit** phục vụ quay video demo (3–5 phút) và thuyết trình.

---

## 2. Công Nghệ & Thư Viện Sử Dụng

- **Ngôn ngữ**: Python 3.10+ (Đã kiểm thử tương thích hoàn toàn trên Python 3.14)
- **Thư viện chính**:
  - `opencv-python`: Cài đặt các thuật toán lọc không gian tối ưu (`cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.bilateralFilter`).
  - `numpy`: Xử lý ma trận dữ liệu ảnh và sinh phân phối nhiễu ngẫu nhiên.
  - `scikit-image`: Đo lường chính xác các chỉ số định lượng PSNR và SSIM đa kênh.
  - `streamlit`: Xây dựng ứng dụng web tương tác trực quan cao.
  - `matplotlib` & `pandas`: Trực quan hóa dữ liệu, vẽ biểu đồ cột và quản lý bảng số liệu.

---

## 3. Cấu Trúc Project

```text
A2XULYANH/
├── app.py                      # Điểm khởi chạy ứng dụng từ thư mục gốc
├── requirements.txt            # Danh sách thư viện cần thiết
├── README.md                   # Tài liệu hướng dẫn kỹ thuật chi tiết
│
├── KhuNhieuAnh/                # THƯ MỤC MÃ NGUỒN CHÍNH
│   ├── ung_dung.py             # Giao diện Streamlit chính (tương tác, biểu đồ, zoom ROI)
│   ├── bo_loc.py               # Cài đặt 4 bộ lọc OpenCV & đo thời gian
│   ├── nhieu.py                # Hàm sinh Nhiễu Gauss & Nhiễu Muối Tiêu
│   ├── chi_so.py               # Hàm tính PSNR, SSIM & xuất báo cáo số liệu
│   ├── thuc_nghiem.py          # Script tự động chạy batch test trên bộ ảnh và xuất CSV
│   ├── tao_anh_mau.py          # Script quản lý và chuẩn bị ảnh mẫu
│   ├── requirements.txt        # File cấu hình thư viện cục bộ
│   │
│   ├── anh/                    # Bộ ảnh thực nghiệm chuẩn (400x400)
│   │   ├── con_cho.jpg         # Ảnh con chó (Đặc trưng lông & chi tiết viền)
│   │   ├── chan_dung.jpg       # Ảnh chân dung (Đặc trưng khuôn mặt, tóc, da)
│   │   ├── phong_canh.jpg      # Ảnh phong cảnh / kiến trúc (Đường nét góc cạnh)
│   │   ├── vat_the.jpg         # Ảnh vật thể (Màu sắc, quả, hình khối)
│   │   └── mau.jpg             # Ảnh mẫu đồ họa gradient
│   │
│   └── ket_qua/                # Thư mục lưu kết quả thực nghiệm tự động
│       ├── ket_qua_thuc_nghiem.csv  # Bảng số liệu PSNR/SSIM/thời gian
│       └── ...                 # Các thư mục chứa ảnh sau khi lọc
```

---

## 4. Mô Tả Lý Thuyết 4 Bộ Lọc

| Bộ lọc | Hàm OpenCV | Nguyên lý toán học | Ưu điểm | Nhược điểm / Hạn chế | Trường hợp tối ưu |
|---|---|---|---|---|---|
| **1. Mean Filter (Trung bình)** | `cv2.blur(src, (k, k))` | Thay pixel bằng trung bình cộng các pixel trong lân cận $k \times k$:<br>$I'(x,y) = \frac{1}{k^2}\sum I(i,j)$ | Rất đơn giản, tốc độ xử lý nhanh nhất | Làm nhòe nặng mọi đường biên cạnh, mất nét chi tiết | Làm mờ hậu cảnh hoặc khử nhiễu thô giá rẻ |
| **2. Gaussian Filter (Gauss)** | `cv2.GaussianBlur(src, (k, k), sigmaX, sigmaY)` | Dùng trọng số Gauss 2D; pixel càng gần tâm có trọng số càng lớn:<br>$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$ | Làm mịn tự nhiên, triệt tiêu nhiễu Gauss tốt hơn Mean | Vẫn là bộ lọc tuyến tính nên làm mờ biên cạnh sắc nét | Khử nhiễu nhẹ liên tục trước khi dò biên |
| **3. Median Filter (Trung vị)** | `cv2.medianBlur(src, k)` | Sắp xếp tất cả pixel lân cận và lấy giá trị nằm ở giữa (trung vị) | **Loại bỏ triệt để điểm ngoại lai (0 và 255)**, giữ cạnh dạng bậc thang rất tốt | Chậm hơn Mean/Gauss, làm tròn các góc nhọn nhỏ | **Số 1 cho Nhiễu Muối Tiêu (Salt & Pepper)** |
| **4. Bilateral Filter (Song phương)** | `cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)` | Kết hợp 2 hàm Gauss: khoảng cách không gian ($c$) và chênh lệch màu ($s$):<br>$BF = \frac{1}{W_p}\sum I_q e^{-\frac{\|p-q\|^2}{2\sigma_s^2}} e^{-\frac{\|I_p-I_q\|^2}{2\sigma_c^2}}$ | **Bảo toàn hoàn hảo đường biên cạnh** trong khi vẫn làm mịn vùng phẳng | Thời gian tính toán cao nhất trong 4 bộ lọc | Khử nhiễu ảnh chân dung, làm mịn da giữ nét mắt/môi |

---

## 5. Mô Tả 2 Loại Nhiễu

1. **Nhiễu Gauss (Gaussian Noise)**:
   - **Bản chất**: Nhiễu cộng liên tục tuân theo phân phối chuẩn $\mathcal{N}(0, \sigma^2)$.
   - **Nguồn gốc**: Nhiễu nhiệt từ cảm biến bán dẫn (sensor noise), chụp ảnh trong điều kiện thiếu sáng hoặc độ nhạy ISO cao.
   - **Công thức mô phỏng**: $I_{noisy}(x,y) = \text{clip}(I(x,y) + \mathcal{N}(0, \sigma^2), 0, 255)$.

2. **Nhiễu Muối Tiêu (Salt & Pepper Noise)**:
   - **Bản chất**: Nhiễu xung rời rạc ngẫu nhiên làm pixel bị biến thành cực đại (Muối: 255 - trắng) hoặc cực tiểu (Tiêu: 0 - đen).
   - **Nguồn gốc**: Lỗi truyền tín hiệu qua kênh số, lỗi bộ chuyển đổi tương tự-số (ADC), hoặc điểm ảnh chết trên cảm biến (dead pixels).

---

## 6. Các Chỉ Số Đánh Giá Định Lượng

1. **PSNR (Peak Signal-to-Noise Ratio)**:
   $$\text{MSE} = \frac{1}{M \times N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I_{goc}(x,y) - I_{loc}(x,y)]^2$$
   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right) \quad (\text{dB})$$
   - *Ý nghĩa*: Giá trị PSNR càng cao chứng tỏ ảnh sau lọc càng gần với ảnh gốc sạch (thường $> 30\text{ dB}$ là chất lượng rất tốt).

2. **SSIM (Structural Similarity Index)**:
   $$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   - *Ý nghĩa*: Đo lường sự tương đồng về độ sáng, độ tương phản và cấu trúc bề mặt. Giá trị nằm trong khoảng $[0, 1]$, càng gần 1.0 nghĩa là cấu trúc ảnh càng được bảo toàn nguyên vẹn.

3. **Thời gian xử lý (Execution Time)**:
   - Đo bằng hàm `time.perf_counter()` riêng cho từng bộ lọc, tính theo mili-giây (ms).

---

## 7. Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 7.1. Cài đặt môi trường
Mở Terminal hoặc PowerShell tại thư mục dự án:
```bash
pip install -r requirements.txt
```

### 7.2. Chạy ứng dụng giao diện Web Streamlit
Từ thư mục gốc `A2XULYANH`:
```bash
streamlit run app.py
```
Hoặc từ thư mục `KhuNhieuAnh`:
```bash
cd KhuNhieuAnh
streamlit run ung_dung.py
```
Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

### 7.3. Chạy thực nghiệm tự động xuất báo cáo (Batch Experiment)
Để tạo bảng số liệu thực nghiệm trên toàn bộ tập ảnh phục vụ viết báo cáo và slide:
```bash
python KhuNhieuAnh/thuc_nghiem.py
```
Kết quả sẽ được xuất ra file CSV tại: `KhuNhieuAnh/ket_qua/ket_qua_thuc_nghiem.csv`.

---

## 8. Kịch Bản Quay Video Demo (3–5 Phút)

- **Bước 1 (30s)**: Mở giao diện Streamlit, giới thiệu đề tài Chủ đề 6 và chọn **Ảnh con chó** (`con_cho.jpg`).
- **Bước 2 (60s) - Thí nghiệm 1 (Nhiễu Gauss)**:
  - Chọn `Nhiễu Gauss (Gaussian Noise)`, để $\sigma = 25$.
  - Nhấn nút **"🚀 Chạy Thực Nghiệm Khử Nhiễu"**.
  - Cho người xem thấy 4 kết quả ảnh sau lọc.
  - Kéo xuống mục **"3. Phóng to vùng chi tiết (ROI Zoom)"** để soi đường biên tai/lông chó: chỉ ra **Bilateral Filter giữ viền cạnh sắc nét** trong khi **Mean/Gauss bị mờ nhòe**.
  - Phân tích bảng số liệu PSNR/SSIM và biểu đồ.
- **Bước 3 (60s) - Thí nghiệm 2 (Nhiễu Muối Tiêu)**:
  - Chuyển sang chọn `Nhiễu Muối Tiêu (Salt & Pepper Noise)`, tỷ lệ 10%.
  - Nhấn nút **"🚀 Chạy Thực Nghiệm Khử Nhiễu"**.
  - So sánh kết quả: Chỉ ra **Median Filter xóa sạch 100% hạt muối tiêu** (PSNR vượt trội ~35 dB), trong khi Bilateral và Mean/Gauss bị nhòe loang lổ.
  - Soi vùng zoom ROI để chứng minh.
- **Bước 4 (30s) - Kết luận**:
  - Đọc phần kết luận khoa học tự động được tạo ở cuối trang web: Mỗi bộ lọc có ưu thế riêng và việc chọn bộ lọc phải phụ thuộc vào loại nhiễu và yêu cầu bảo toàn biên.
