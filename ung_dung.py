"""
ung_dung.py (Root wrapper) - Cho phép chạy trực tiếp 'streamlit run ung_dung.py' ngay từ thư mục gốc.
"""
import os
import sys

thu_muc_goc = os.path.dirname(os.path.abspath(__file__))
thu_muc_code = os.path.join(thu_muc_goc, "KhuNhieuAnh")

if thu_muc_code not in sys.path:
    sys.path.insert(0, thu_muc_code)

duong_dan_ung_dung = os.path.join(thu_muc_code, "ung_dung.py")

with open(duong_dan_ung_dung, "r", encoding="utf-8") as f:
    code_content = f.read()

exec(compile(code_content, duong_dan_ung_dung, "exec"))
