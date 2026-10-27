import streamlit as st
import cv2
import numpy as np

# 1. Module Biến đổi mức xám & Tương phản
from modules.point_processing import (
    apply_negative,
    apply_log,
    apply_gamma,
    apply_hist_equalization,
    apply_clahe
)

# 2. Module Trích xuất biên cạnh
from modules.edge_detection import (
    apply_sobel,
    apply_prewitt,
    apply_canny
)

# 3. Module Lọc không gian
from modules.spatial_filtering import (
    apply_gaussian,
    apply_median
)

st.set_page_config(
    page_title="Ứng Dụng Xử Lý Ảnh Số - Nhóm 3",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS: RETRO EDITORIAL THEME =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #f7f3ea !important;
        color: #1e293b !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #ede7d8 !important;
        border-right: 2px solid #242424 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }

    .app-header {
        border-bottom: 2px solid #242424;
        padding-bottom: 14px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .app-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: #1e293b;
    }
    .app-desc {
        font-size: 0.88rem;
        color: #57534e;
        margin-top: 4px;
    }

    /* Thẻ chữ nhật bo góc viền đen đậm cho menu chức năng */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #fcfbf9 !important;
        border: 1.5px solid #242424 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 8px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin: 0 !important;
    }
    /* Thẻ được chọn: Nền cam đất, chữ trắng */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #f26430 !important;
        border: 2px solid #242424 !important;
        box-shadow: 2px 2px 0px #242424 !important;
        transform: translate(-1px, -1px);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .upload-frame {
        border: 2px dashed #242424;
        border-radius: 12px;
        background: #fdfbf7;
        padding: 24px;
        text-align: center;
        margin-bottom: 12px;
    }

    /* Khung tải file */
    [data-testid="stFileUploader"] {
        background-color: #fcfbf9 !important;
        border: 2px solid #242424 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        box-shadow: 3px 3px 0px #242424 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #f26430 !important;
        color: #ffffff !important;
        border: 1.5px solid #242424 !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        box-shadow: 2px 2px 0px #242424 !important;
    }

    /* Nút gạt Toggle Switch */
    [data-testid="stToggle"] {
        padding: 6px 0;
    }
    [data-testid="stToggle"] label p {
        font-weight: 700 !important;
        color: #0f172a !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stToggle"] span[role="checkbox"] {
        background-color: #cbd5e1 !important;
        border: 1.5px solid #242424 !important;
    }
    [data-testid="stToggle"] span[role="checkbox"][aria-checked="true"] {
        background-color: #f26430 !important;
    }

    .preview-card {
        background: #fdfbf7;
        border: 2px solid #242424;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 10px;
    }
    .preview-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        justify-content: space-between;
    }

    .status-stamp {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        background: #e7dfcc;
        color: #1e293b;
        border: 1.5px solid #242424;
    }

    .stDownloadButton button, .stButton button {
        background-color: #f26430 !important;
        color: #ffffff !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        border: 2px solid #242424 !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        box-shadow: 2px 2px 0px #242424 !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        transform: translate(-1px, -1px) !important;
        box-shadow: 4px 4px 0px #242424 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN TỰ ĐỘNG THÔNG SỐ (AUTO-PILOT) ---
def auto_gamma_correction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    mean_val = np.mean(gray)
    if mean_val < 120:
        auto_gamma = max(0.4, float(mean_val / 140.0))
    elif mean_val > 140:
        auto_gamma = min(1.8, float(mean_val / 110.0))
    else:
        auto_gamma = 1.0
    return apply_gamma(img, auto_gamma), auto_gamma

def auto_canny_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return apply_canny(img, lower, upper), lower, upper

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
        <div style="padding: 5px 0 12px 0; border-bottom: 2px solid #242424; margin-bottom: 12px;">
            <div style="font-family: 'Space Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #0f172a;">
                [ BẢNG ĐIỀU KHIỂN ]
            </div>
            <div style="font-size: 0.78rem; color: #57534e;">Chọn tác vụ xử lý</div>
        </div>
    """, unsafe_allow_html=True)

    feature = st.radio(
        label="Chức năng",
        options=[
            "1. Làm sáng vùng tối & Cứu ảnh ngược sáng",
            "2. Khử sương mờ & Tăng độ tương phản",
            "3. Khử nhiễu hạt & Làm mịn bề mặt ảnh",
            "4. Chuyển đổi ảnh sang dạng âm bản",
            "5. Trích xuất đường nét & Khung viền"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    advanced_mode = st.toggle("Bật tinh chỉnh thủ công", value=False)

# ================= MÀN HÌNH CHÍNH =================
st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">ỨNG DỤNG XỬ LÝ & PHỤC HỒI ẢNH SỐ</div>
            <div class="app-desc">Tiền xử lý và tối ưu hóa các khiếm khuyết vật lý dựa trên xử lý điểm ảnh</div>
        </div>
        <div class="status-stamp">NHÓM 3</div>
    </div>
""", unsafe_allow_html=True)

if 'saved_img_bytes' not in st.session_state:
    st.session_state.saved_img_bytes = None

# GIAI ĐOẠN 1: CHƯA NẠP ẢNH
if st.session_state.saved_img_bytes is None:
    st.markdown("""
        <div class="upload-frame">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">🎞️</div>
            <div style="font-family: 'Space Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                NẠP ẢNH CẦN XỬ LÝ
            </div>
            <div style="font-size: 0.85rem; color: #78716c; margin-bottom: 15px;">
                Hỗ trợ ảnh chụp ngược sáng, mờ sương, dính hạt nhiễu hoặc phim âm bản
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Tải ảnh từ máy tính", 
        type=["png", "jpg", "jpeg", "bmp"],
        key="main_uploader",
        label_visibility="collapsed"
    )
    if uploaded is not None:
        st.session_state.saved_img_bytes = uploaded.getvalue()
        st.rerun()

# GIAI ĐOẠN 2: ĐÃ CÓ ẢNH TRONG BỘ NHỚ
else:
    file_bytes = np.frombuffer(st.session_state.saved_img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result_img = None
    badge_info = ""

    # Chức năng 1: Cứu sáng
    if feature == "1. Làm sáng vùng tối & Cứu ảnh ngược sáng":
        if not advanced_mode:
            result_img, calculated_gamma = auto_gamma_correction(img_rgb)
            badge_info = f"AUTO-GAMMA: {calculated_gamma:.2f}"
        else:
            with st.sidebar:
                m_type = st.radio("Phương thức:", ["Hiệu chỉnh Gamma", "Biến đổi Logarit"])
                if m_type == "Hiệu chỉnh Gamma":
                    g_val = st.slider("Hệ số Gamma (γ < 1 làm sáng, γ > 1 dằn chói):", 0.1, 2.5, 0.6, 0.05)
                    result_img = apply_gamma(img_rgb, g_val)
                    badge_info = f"MANUAL-GAMMA: {g_val}"
                else:
                    result_img = apply_log(img_rgb)
                    badge_info = "LOGARIT TRANSFORM"

    # Chức năng 2: Sương mù & Tương phản (CLAHE)
    elif feature == "2. Khử sương mờ & Tăng độ tương phản":
        if not advanced_mode:
            result_img = apply_hist_equalization(img_rgb)
            badge_info = "HISTOGRAM EQUALIZATION (GLOBAL)"
        else:
            with st.sidebar:
                st.markdown("<p style='font-size: 0.82rem; font-weight: 700; margin-bottom: 4px;'>CẤU HÌNH BỘ LỌC CLAHE:</p>", unsafe_allow_html=True)
                c_limit = st.slider("Mức độ khử sương (Clip Limit):", 1.0, 8.0, 3.0, 0.5)
                g_size = st.select_slider("Kích thước lưới chia (Tile Grid):", options=[4, 8, 16], value=8)
                
                result_img = apply_clahe(img_rgb, clip_limit=c_limit, grid_size=g_size)
                badge_info = f"CLAHE (Clip: {c_limit}, Grid: {g_size}x{g_size})"

    # Chức năng 3: Khử hạt
    elif feature == "3. Khử nhiễu hạt & Làm mịn bề mặt ảnh":
        if not advanced_mode:
            result_img = apply_median(img_rgb, 3)
            badge_info = "AUTO MEDIAN (3x3)"
        else:
            with st.sidebar:
                n_algo = st.radio("Bộ lọc:", ["Lọc trung vị (Median)", "Lọc Gauss (Gaussian)"])
                k_val = st.slider("Kích thước Kernel:", 3, 15, 3, step=2)
                if n_algo == "Lọc trung vị (Median)":
                    result_img = apply_median(img_rgb, k_val)
                else:
                    result_img = apply_gaussian(img_rgb, k_val)
                badge_info = f"{n_algo.upper()} ({k_val}x{k_val})"

    # Chức năng 4: Âm bản
    elif feature == "4. Chuyển đổi ảnh sang dạng âm bản":
        result_img = apply_negative(img_rgb)
        badge_info = "NEGATIVE (s = 255 - r)"

    # Chức năng 5: Bóc tách khung viền
    elif feature == "5. Trích xuất đường nét & Khung viền":
        if not advanced_mode:
            result_img, low, high = auto_canny_edge(img_rgb)
            badge_info = f"CANNY EDGE: [{low}, {high}]"
        else:
            with st.sidebar:
                edge_op = st.radio("Toán tử:", ["Canny Edge", "Sobel Operator", "Prewitt Operator"])
                if edge_op == "Canny Edge":
                    l_val = st.slider("Ngưỡng dưới (Low):", 0, 255, 50)
                    h_val = st.slider("Ngưỡng trên (High):", 0, 255, 150)
                    result_img = apply_canny(img_rgb, l_val, h_val)
                    badge_info = f"CANNY: [{l_val}, {h_val}]"
                elif edge_op == "Sobel Operator":
                    result_img = apply_sobel(img_rgb, 3)
                    badge_info = "SOBEL (3x3)"
                else:
                    result_img = apply_prewitt(img_rgb)
                    badge_info = "PREWITT (3x3)"

    # Hiển thị 2 cột Before / After
    col_before, col_after = st.columns(2, gap="medium")

    with col_before:
        st.markdown(f"""
            <div class="preview-card">
                <div class="preview-header">
                    <span>[ ẢNH BAN ĐẦU ]</span>
                    <span>{img_rgb.shape[1]}x{img_rgb.shape[0]} PX</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with col_after:
        st.markdown(f"""
            <div class="preview-card">
                <div class="preview-header">
                    <span>[ KẾT QUẢ XỬ LÝ ]</span>
                    <span class="status-stamp">{badge_info}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.image(result_img, use_container_width=True)

    # KHU VỰC ĐIỀU KHIỂN DƯỚI ẢNH
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    ctrl_col1, ctrl_col2 = st.columns([1, 1], gap="medium")

    with ctrl_col1:
        if st.button("🔄 CHỌN BỨC ẢNH KHÁC", use_container_width=True):
            st.session_state.saved_img_bytes = None
            st.rerun()

    with ctrl_col2:
        out_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", out_bgr)
        st.download_button(
            label="💾 TẢI ẢNH KẾT QUẢ VỀ MÁY (.PNG)",
            data=buffer.tobytes(),
            file_name="anh_da_xu_ly.png",
            mime="image/png",
            use_container_width=True
        )