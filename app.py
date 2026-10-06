"""
app.py - Điểm khởi chạy chính ở thư mục gốc của project.
Chủ đề 6: Các bộ lọc xử lý ảnh & So sánh khả năng khử nhiễu.

Chạy: streamlit run app.py
hoặc: streamlit run KhuNhieuAnh/ung_dung.py
"""

import os
import sys

# Đảm bảo đường dẫn module luôn tìm thấy trong KhuNhieuAnh
thu_muc_goc = os.path.dirname(os.path.abspath(__file__))
thu_muc_code = os.path.join(thu_muc_goc, "KhuNhieuAnh")

if thu_muc_code not in sys.path:
    sys.path.insert(0, thu_muc_code)

# Đổi thư mục làm việc hoặc thực thi module ung_dung.py
duong_dan_ung_dung = os.path.join(thu_muc_code, "ung_dung.py")

with open(duong_dan_ung_dung, "r", encoding="utf-8") as f:
    code_content = f.read()

exec(compile(code_content, duong_dan_ung_dung, "exec"))
