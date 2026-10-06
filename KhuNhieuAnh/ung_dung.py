"""
ung_dung.py – Web App xử lý ảnh | Chủ đề 6: So sánh 4 bộ lọc khử nhiễu
Giao diện: Light Dashboard hiện đại, nền sáng, chữ đen đậm rõ nét, chuẩn báo cáo học thuật.

Chạy: streamlit run ung_dung.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Module path ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nhieu import them_nhieu_gauss, them_nhieu_muoi_tieu
from bo_loc import (
    loc_trung_binh, loc_gauss, loc_trung_vi, loc_song_phuong,
    TEN_TRUNG_BINH, TEN_GAUSS, TEN_TRUNG_VI, TEN_SONG_PHUONG,
)
from chi_so import tinh_psnr, tinh_ssim

# ── Image folder (multi-fallback) ───────────────────────────────────────────
def _find_anh_dir() -> str:
    candidates = [
        os.path.join(_HERE, "anh"),
        os.path.join(_HERE, "KhuNhieuAnh", "anh"),
        os.path.join(os.getcwd(), "anh"),
        os.path.join(os.getcwd(), "KhuNhieuAnh", "anh"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

ANH_DIR = _find_anh_dir()

SAMPLE_IMAGES = {
    "con_cho.jpg":    "🐶 Ảnh con chó (Đặc trưng lông & biên viền)",
    "chan_dung.jpg":  "👤 Ảnh chân dung (Khuôn mặt & chi tiết tóc)",
    "phong_canh.jpg": "🏛️ Ảnh phong cảnh (Kiến trúc & đường nét)",
    "vat_the.jpg":    "🍎 Ảnh vật thể (Khối màu & quả)",
    "mau.jpg":        "🎨 Ảnh mẫu hình học (Đồ họa & gradient)",
}

FILTER_META = {
    TEN_TRUNG_BINH:  {"en": "Mean Filter",      "icon": "🔵", "fn": "cv2.blur()"},
    TEN_GAUSS:       {"en": "Gaussian Filter",   "icon": "🟢", "fn": "cv2.GaussianBlur()"},
    TEN_TRUNG_VI:    {"en": "Median Filter",     "icon": "🟠", "fn": "cv2.medianBlur()"},
    TEN_SONG_PHUONG: {"en": "Bilateral Filter",  "icon": "🟣", "fn": "cv2.bilateralFilter()"},
}

COLORS = {
    TEN_TRUNG_BINH:  "#2563eb",   # blue
    TEN_GAUSS:       "#16a34a",   # green
    TEN_TRUNG_VI:    "#ea580c",   # orange
    TEN_SONG_PHUONG: "#9333ea",   # purple
}

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Khử Nhiễu Ảnh Số - Chủ Đề 6",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS – Giao diện Sáng (Light Theme), Chữ đen rõ ràng, Chuẩn Báo Cáo
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Nền trang & Màu chữ toàn cục ─── */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* ─── Thanh Sidebar bên trái ─── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.03) !important;
}
[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* ─── Header Banner chính ─── */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
    border-radius: 12px;
    padding: 24px 30px 20px;
    margin-bottom: 22px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.18);
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.app-header .subtitle {
    color: #dbeafe !important;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.8px;
}
.app-header .badge-row {
    margin-top: 14px;
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255, 255, 255, 0.22);
    border: 1px solid rgba(255, 255, 255, 0.45);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #ffffff !important;
    font-weight: 600;
}

/* ─── Tiêu đề các mục lớn ─── */
.section-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #1e3a8a !important;
    border-left: 4px solid #2563eb;
    padding-left: 10px;
    margin: 22px 0 14px 0;
    letter-spacing: 0.2px;
}

/* ─── Tiêu đề card & hàm opencv ─── */
.card-header-title {
    font-size: 0.96rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 2px;
    text-align: center;
}
.card-header-fn {
    font-size: 0.74rem;
    color: #64748b;
    font-family: Consolas, monospace;
    margin-bottom: 8px;
    text-align: center;
}

/* ─── Metric pills bên trong card ─── */
.metric-row {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.metric-pill {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #1e293b;
    font-weight: 600;
}
.metric-pill span {
    color: #0f172a;
    font-weight: 700;
}
.metric-pill.best {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #065f46;
}
.metric-pill.best span {
    color: #047857;
}

/* ─── Bảng so sánh HTML chuẩn ─── */
.compare-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 10px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.compare-table th {
    background: #f1f5f9;
    color: #0f172a;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 11px 16px;
    text-align: left;
    border-bottom: 2px solid #cbd5e1;
}
.compare-table td {
    color: #1e293b;
    font-size: 0.88rem;
    padding: 10px 16px;
    border-bottom: 1px solid #f1f5f9;
}
.compare-table tr:last-child td {
    border-bottom: none;
}
.compare-table tr:hover td {
    background: #f8fafc;
}
.best-row td {
    color: #047857 !important;
    font-weight: 700;
    background: #f0fdf4 !important;
}

/* ─── Hộp trạng thái ban đầu ─── */
.status-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 20px 24px;
    color: #1e3a8a;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ─── Phân cách sidebar ─── */
.sidebar-section {
    font-size: 0.78rem;
    font-weight: 800;
    color: #1d4ed8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 18px 0 8px 0;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
}

/* ─── Nút bấm chính ─── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
}

/* ─── Khung Upload File (Ép sáng, không còn bị đen xì) ─── */
[data-testid="stFileUploader"] {
    background-color: #f8fafc !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    border: 1px dashed #94a3b8 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] section * {
    color: #0f172a !important;
}
[data-testid="stFileUploader"] button {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

/* ─── Expander Tham số (Ép nền sáng, chữ đen) ─── */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #0f172a !important;
}

/* ─── Container viền bo góc nền trắng ─── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    padding: 8px !important;
}

/* Ẩn footer mặc định */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def load_sample(filename: str) -> np.ndarray | None:
    path = os.path.join(ANH_DIR, filename)
    if os.path.exists(path):
        return cv2.imread(path)
    return None


def load_upload(file_obj) -> np.ndarray | None:
    if file_obj is None:
        return None
    arr = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def roi_zoom(img: np.ndarray, cx_pct: float, cy_pct: float,
             size: int = 80, scale: int = 3) -> tuple:
    h, w = img.shape[:2]
    cx = int(w * cx_pct / 100)
    cy = int(h * cy_pct / 100)
    half = size // 2
    x1 = max(0, min(cx - half, w - size))
    y1 = max(0, min(cy - half, h - size))
    x2 = x1 + size
    y2 = y1 + size
    crop = img[y1:y2, x1:x2]
    zoomed = cv2.resize(crop, (size * scale, size * scale),
                        interpolation=cv2.INTER_NEAREST)
    return zoomed, (x1, y1, x2, y2)


def draw_roi_box(img: np.ndarray, box: tuple) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


def make_bar_chart(labels, values, title, ylabel, color_list):
    """Vẽ biểu đồ cột nền trắng, chữ đen sắc nét cho bài báo cáo/slide."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    bars = ax.bar(labels, values, color=color_list, width=0.45,
                  edgecolor="#cbd5e1", linewidth=0.8, zorder=3)
    ax.set_title(title, color="#0f172a", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color="#334155", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#334155", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    max_val = max(values) if values and max(values) > 0 else 1.0
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015 * max_val,
                f"{v:.2f}", ha="center", va="bottom", color="#0f172a", fontsize=8.5,
                fontweight="bold")
    plt.tight_layout(pad=0.8)
    return fig


def run_filter(name: str, noisy: np.ndarray, ksize: int,
               sigma_g: float, d_bi: int, sc: float, ss: float):
    if name == TEN_TRUNG_BINH:
        return loc_trung_binh(noisy, ksize)
    elif name == TEN_GAUSS:
        return loc_gauss(noisy, ksize, sigma_g)
    elif name == TEN_TRUNG_VI:
        return loc_trung_vi(noisy, ksize)
    elif name == TEN_SONG_PHUONG:
        return loc_song_phuong(noisy, d_bi, sc, ss)
    return noisy.copy(), 0.0


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
defaults = dict(
    done=False, orig=None, noisy=None,
    results={}, metrics={}, noise_label="",
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🖼️ Ứng dụng Xử lý Ảnh – Giảm Nhiễu &amp; So Sánh Bộ Lọc</h1>
  <div class="subtitle">Mean &nbsp;|&nbsp; Gaussian &nbsp;|&nbsp; Median &nbsp;|&nbsp; Bilateral Filter</div>
  <div class="badge-row">
    <span class="badge">Python · OpenCV</span>
    <span class="badge">Streamlit</span>
    <span class="badge">scikit-image</span>
    <span class="badge">NumPy</span>
    <span class="badge">Chủ đề 6 · Xử lý ảnh số</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Bảng Điều Khiển")

    # ── 1. Chọn ảnh ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">1 · Chọn ảnh</div>', unsafe_allow_html=True)
    img_source = st.radio("Nguồn ảnh:", ["Ảnh mẫu có sẵn", "Upload ảnh"],
                          horizontal=True, label_visibility="collapsed")

    img_raw: np.ndarray | None = None
    img_label = ""

    if img_source == "Ảnh mẫu có sẵn":
        chosen = st.selectbox(
            "Chọn ảnh mẫu:",
            options=list(SAMPLE_IMAGES.keys()),
            format_func=lambda x: SAMPLE_IMAGES[x],
        )
        img_raw = load_sample(chosen)
        img_label = chosen
        if img_raw is not None:
            st.image(bgr2rgb(img_raw),
                     caption=f"✅ {chosen} ({img_raw.shape[1]}×{img_raw.shape[0]} px)",
                     use_container_width=True)
        else:
            st.error(f"❌ Không tìm thấy `{chosen}` trong `{ANH_DIR}`")
    else:
        uploaded = st.file_uploader("Chọn file ảnh (JPG, PNG, BMP):", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            img_raw = load_upload(uploaded)
            img_label = uploaded.name
            if img_raw is not None:
                st.image(bgr2rgb(img_raw),
                         caption=f"✅ {uploaded.name}",
                         use_container_width=True)

    # ── 2. Loại nhiễu ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">2 · Loại nhiễu</div>', unsafe_allow_html=True)
    noise_type = st.radio("Loại nhiễu:", ["Gaussian Noise", "Salt & Pepper Noise"],
                          label_visibility="collapsed")

    if noise_type == "Gaussian Noise":
        noise_sigma = st.slider("Sigma (độ mạnh nhiễu)", 5, 80, 25, 5,
                                help="Giá trị lớn → nhiễu mạnh hơn")
    else:
        noise_ratio = st.slider("Tỷ lệ nhiễu (%)", 1, 40, 10, 1,
                                help="% pixels bị biến thành trắng/đen")

    # ── 3. Chọn bộ lọc ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">3 · Bộ lọc áp dụng</div>', unsafe_allow_html=True)
    use_mean      = st.checkbox("Mean Filter (Trung bình)",      value=True)
    use_gaussian  = st.checkbox("Gaussian Filter (Gauss)",       value=True)
    use_median    = st.checkbox("Median Filter (Trung vị)",      value=True)
    use_bilateral = st.checkbox("Bilateral Filter (Song phương)", value=True)

    selected_filters = []
    if use_mean:      selected_filters.append(TEN_TRUNG_BINH)
    if use_gaussian:  selected_filters.append(TEN_GAUSS)
    if use_median:    selected_filters.append(TEN_TRUNG_VI)
    if use_bilateral: selected_filters.append(TEN_SONG_PHUONG)

    # ── Tham số bộ lọc ────────────────────────────────────────────────────
    with st.expander("🔧 Tham số chi tiết bộ lọc", expanded=False):
        ksize   = st.slider("Kernel size (k×k)", 3, 15, 5, 2)
        sigma_g = st.slider("Gaussian σ", 0.5, 5.0, 1.5, 0.5)
        st.markdown("**Bilateral Filter:**")
        d_bi    = st.slider("d (đường kính lân cận)", 3, 15, 9, 2)
        sigma_c = st.slider("σ_color (giữ biên màu)", 10, 150, 75, 5)
        sigma_s = st.slider("σ_space (không gian)", 10, 150, 75, 5)

    st.markdown("")

    # ── 4. Nút chạy ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">4 · Thực hiện</div>', unsafe_allow_html=True)
    btn_run   = st.button("▶  Bắt đầu xử lý", use_container_width=True)
    btn_reset = st.button("↺  Đặt lại", use_container_width=True)

    if btn_reset:
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PROCESSING
# ════════════════════════════════════════════════════════════════════════════
if btn_run:
    if img_raw is None:
        st.warning("⚠️ Vui lòng chọn hoặc tải ảnh trước khi xử lý.")
    elif not selected_filters:
        st.warning("⚠️ Vui lòng chọn ít nhất một bộ lọc.")
    else:
        with st.spinner("⏳ Đang xử lý các bộ lọc..."):
            # Tạo nhiễu
            if noise_type == "Gaussian Noise":
                noisy = them_nhieu_gauss(img_raw, float(noise_sigma))
                noise_label = f"Gaussian Noise (σ = {noise_sigma})"
            else:
                noisy = them_nhieu_muoi_tieu(img_raw, float(noise_ratio) / 100.0)
                noise_label = f"Salt & Pepper Noise ({noise_ratio}%)"

            # Chạy từng bộ lọc được chọn
            results = {}
            metrics = {}
            for name in selected_filters:
                filtered, t = run_filter(name, noisy, ksize, sigma_g, d_bi, sigma_c, sigma_s)
                psnr = tinh_psnr(img_raw, filtered)
                ssim = tinh_ssim(img_raw, filtered)
                results[name] = filtered
                metrics[name] = {
                    "psnr": psnr,
                    "ssim": ssim,
                    "time_ms": round(t * 1000, 2),
                }

        st.session_state.update(
            done=True, orig=img_raw, noisy=noisy,
            results=results, metrics=metrics, noise_label=noise_label,
        )


# ════════════════════════════════════════════════════════════════════════════
# RESULTS AREA
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.done:
    # Trạng thái ban đầu
    st.markdown("""
    <div class="status-box">
      👈 <strong>Hướng dẫn thực hiện:</strong><br>
      1. <b>Chọn ảnh mẫu</b> hoặc tải ảnh từ máy tính ở bảng điều khiển bên trái.<br>
      2. <b>Chọn loại nhiễu</b> (Gaussian hoặc Muối Tiêu) và điều chỉnh mức độ nhiễu.<br>
      3. <b>Chọn các bộ lọc</b> cần so sánh.<br>
      4. Bấm <b>▶ Bắt đầu xử lý</b> để chạy thực nghiệm và hiển thị bảng so sánh.<br><br>
      Toàn bộ kết quả PSNR, SSIM và thời gian xử lý được tính toán thực tế từ OpenCV pipeline.
    </div>
    """, unsafe_allow_html=True)

else:
    orig        = st.session_state.orig
    noisy       = st.session_state.noisy
    results     = st.session_state.results
    metrics     = st.session_state.metrics
    noise_label = st.session_state.noise_label

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 1 – Ảnh gốc & Ảnh nhiễu
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📷 Ảnh Gốc &amp; Ảnh Sau Khi Thêm Nhiễu</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="card-header-title">Ảnh Gốc (Clean)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-header-fn">{orig.shape[1]}×{orig.shape[0]} px · Chuẩn đối chứng</div>', unsafe_allow_html=True)
            st.image(bgr2rgb(orig), use_container_width=True)
    with c2:
        with st.container(border=True):
            noisy_psnr = tinh_psnr(orig, noisy)
            noisy_ssim = tinh_ssim(orig, noisy)
            st.markdown(f'<div class="card-header-title">Ảnh Nhiễu · {noise_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-header-fn">Chất lượng ảnh ban đầu sau khi pha nhiễu</div>', unsafe_allow_html=True)
            st.image(bgr2rgb(noisy), use_container_width=True)
            st.markdown(
                f'<div class="metric-row">'
                f'<span class="metric-pill">PSNR: <span>{noisy_psnr:.2f} dB</span></span>'
                f'<span class="metric-pill">SSIM: <span>{noisy_ssim:.4f}</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 2 – Kết quả 4 bộ lọc (cards viền bo góc)
    # ──────────────────────────────────────────────────────────────────────
    if results:
        st.markdown('<div class="section-title">🔍 Kết Quả Khử Nhiễu Các Bộ Lọc</div>',
                    unsafe_allow_html=True)

        best_filter = max(metrics, key=lambda n: metrics[n]["psnr"])

        cols = st.columns(len(results), gap="small")
        for col, name in zip(cols, results.keys()):
            m = metrics[name]
            meta = FILTER_META[name]
            is_best = (name == best_filter)
            pill_cls = "metric-pill best" if is_best else "metric-pill"
            best_tag = " 🏆 (Tốt nhất)" if is_best else ""

            with col:
                with st.container(border=True):
                    st.markdown(f'<div class="card-header-title">{meta["icon"]} {meta["en"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-header-fn">{meta["fn"]}</div>', unsafe_allow_html=True)
                    st.image(bgr2rgb(results[name]), use_container_width=True)
                    st.markdown(f"""
                      <div class="metric-row">
                        <span class="{pill_cls}">PSNR: <span>{m["psnr"]:.2f} dB{best_tag}</span></span>
                      </div>
                      <div class="metric-row">
                        <span class="{pill_cls}">SSIM: <span>{m["ssim"]:.4f}</span></span>
                        <span class="metric-pill">⏱ <span>{m["time_ms"]:.1f} ms</span></span>
                      </div>
                    """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────
        # SECTION 3 – Bảng so sánh
        # ──────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Bảng So Sánh Định Lượng (PSNR / SSIM / Thời Gian)</div>',
                    unsafe_allow_html=True)

        rows_html = ""
        for name in results:
            m = metrics[name]
            meta = FILTER_META[name]
            is_best = (name == best_filter)
            row_cls = "best-row" if is_best else ""
            tag = " 🏆 (Tốt nhất)" if is_best else ""
            rows_html += f"""
            <tr class="{row_cls}">
              <td><b>{meta["icon"]} {meta["en"]}</b>{tag}</td>
              <td><b>{m["psnr"]:.2f}</b></td>
              <td><b>{m["ssim"]:.4f}</b></td>
              <td>{m["time_ms"]:.2f}</td>
            </tr>"""

        st.markdown(f"""
        <table class="compare-table">
          <thead>
            <tr>
              <th>Bộ Lọc</th>
              <th>PSNR (dB) ↑ (Càng cao càng tốt)</th>
              <th>SSIM ↑ (Gần 1.0 càng tốt)</th>
              <th>Thời Gian (ms) ↓ (Càng thấp càng nhanh)</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────
        # SECTION 4 – Biểu đồ so sánh
        # ──────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📈 Biểu Đồ Trực Quan Hóa Định Lượng</div>',
                    unsafe_allow_html=True)

        names    = list(results.keys())
        en_names = [FILTER_META[n]["en"].replace(" Filter", "") for n in names]
        psnrs    = [metrics[n]["psnr"]    for n in names]
        ssims    = [metrics[n]["ssim"]    for n in names]
        times    = [metrics[n]["time_ms"] for n in names]
        clrs     = [COLORS[n]             for n in names]

        bc1, bc2, bc3 = st.columns(3, gap="medium")
        with bc1:
            fig1 = make_bar_chart(en_names, psnrs, "PSNR (dB) – Cao hơn tốt hơn", "dB", clrs)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
        with bc2:
            fig2 = make_bar_chart(en_names, ssims, "SSIM – Gần 1.0 là tốt nhất", "SSIM", clrs)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        with bc3:
            fig3 = make_bar_chart(en_names, times, "Thời gian xử lý (ms) – Nhanh hơn", "ms", clrs)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        # ──────────────────────────────────────────────────────────────────
        # SECTION 5 – ROI Zoom
        # ──────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔬 Phóng To Vùng Chi Tiết (ROI) 3× – So Sánh Đường Biên &amp; Hạt Nhiễu</div>',
                    unsafe_allow_html=True)
        st.info("💡 Kéo thanh trượt để di chuyển vùng ROI (khung đỏ) và quan sát trực tiếp khả năng **bảo toàn cạnh sắc nét** so với **làm mờ biên**.")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            roi_cx = st.slider("Tọa độ tâm X (%)", 10, 90, 50, 5, key="roi_cx")
        with rc2:
            roi_cy = st.slider("Tọa độ tâm Y (%)", 10, 90, 50, 5, key="roi_cy")
        with rc3:
            roi_sz = st.slider("Kích thước vùng crop (px)", 40, 120, 80, 10, key="roi_sz")

        # Cắt và phóng to
        rz_orig, box = roi_zoom(orig, roi_cx, roi_cy, roi_sz, 3)
        rz_noisy, _  = roi_zoom(noisy, roi_cx, roi_cy, roi_sz, 3)
        orig_marked  = draw_roi_box(orig, box)

        zc0, zc1 = st.columns([1.1, 2.9], gap="medium")
        with zc0:
            with st.container(border=True):
                st.markdown('<div class="card-header-title">Vị Trí ROI (Khung Đỏ)</div>', unsafe_allow_html=True)
                st.image(bgr2rgb(orig_marked), use_container_width=True)
        with zc1:
            ref_cols = st.columns(2)
            with ref_cols[0]:
                with st.container(border=True):
                    st.markdown('<div class="card-header-title">Ảnh Gốc (×3)</div>', unsafe_allow_html=True)
                    st.image(bgr2rgb(rz_orig), use_container_width=True)
            with ref_cols[1]:
                with st.container(border=True):
                    st.markdown('<div class="card-header-title">Ảnh Nhiễu (×3)</div>', unsafe_allow_html=True)
                    st.image(bgr2rgb(rz_noisy), use_container_width=True)

        if results:
            filter_cols = st.columns(len(results), gap="small")
            for col, name in zip(filter_cols, results.keys()):
                rz, _ = roi_zoom(results[name], roi_cx, roi_cy, roi_sz, 3)
                with col:
                    with st.container(border=True):
                        st.markdown(f'<div class="card-header-title">{FILTER_META[name]["icon"]} {FILTER_META[name]["en"]} (×3)</div>', unsafe_allow_html=True)
                        st.image(bgr2rgb(rz), use_container_width=True)

        # ──────────────────────────────────────────────────────────────────
        # SECTION 6 – Nhận xét khoa học
        # ──────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📝 Đánh Giá &amp; Kết Luận Thực Nghiệm</div>',
                    unsafe_allow_html=True)

        df = pd.DataFrame(metrics).T
        best_psnr_name = df["psnr"].idxmax()
        fastest_name   = df["time_ms"].idxmin()

        is_gauss_noise = "Gaussian" in noise_label

        if is_gauss_noise:
            remark = f"""
- **Loại nhiễu thử nghiệm:** **{noise_label}** (Nhiễu phân bố liên tục theo hàm mật độ Gauss $\\mathcal{{N}}(0, \\sigma^2)$).
- **Bộ lọc có chất lượng phục hồi cao nhất:** **{FILTER_META[best_psnr_name]['en']}** với **PSNR = {metrics[best_psnr_name]['psnr']:.2f} dB** và **SSIM = {metrics[best_psnr_name]['ssim']:.4f}**.
- **Đặc trưng biên cạnh:** **Bilateral Filter (Bộ lọc song phương)** nổi bật ở khả năng vừa làm mượt các hạt nhiễu Gauss vừa **bảo toàn các đường biên sắc nét**, không làm mờ nhoè chi tiết mạnh như Mean Filter.
- **Mean Filter & Gaussian Filter:** Có hiệu quả làm mịn tốt nhưng làm mờ các chi tiết tần số cao (mờ lông, viền cạnh).
            """
        else:
            remark = f"""
- **Loại nhiễu thử nghiệm:** **{noise_label}** (Nhiễu xung rời rạc, làm hỏng pixel thành cực trị 0 hoặc 255).
- **Bộ lọc hiệu quả nhất:** **{FILTER_META[best_psnr_name]['en']}** đạt **PSNR = {metrics[best_psnr_name]['psnr']:.2f} dB** và **SSIM = {metrics[best_psnr_name]['ssim']:.4f}**.
- **Nguyên lý:** **Median Filter (Bộ lọc trung vị)** vượt trội hoàn toàn với nhiễu muối tiêu vì giá trị trung vị loại bỏ triệt để các điểm ngoại lai (0 và 255) thay vì lấy trung bình.
- **Hạn chế:** Các bộ lọc tuyến tính (Mean, Gaussian) và Bilateral thất bại trong việc xoá các đốm muối tiêu do phép cộng pixel bị các điểm cực trị kéo lệch nghiêm trọng.
            """

        remark += f"\n- **Tốc độ xử lý:** Bộ lọc **{FILTER_META[fastest_name]['en']}** chạy nhanh nhất ({metrics[fastest_name]['time_ms']:.2f} ms)."

        with st.container(border=True):
            st.markdown(remark)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:0.85rem;padding:6px 0;">'
    '🖼️ Đồ Án Môn Xử Lý Ảnh Số &nbsp;·&nbsp; '
    'Chủ Đề 6: Các Bộ Lọc Xử Lý Ảnh (Mean, Gaussian, Median, Bilateral Filter)'
    '</div>',
    unsafe_allow_html=True,
)
