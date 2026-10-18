import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Phân loại rác thải thông minh",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CẤU HÌNH MÔ HÌNH  (chỉnh theo mô hình của bạn)
# ============================================================
from pathlib import Path

APP_DIR = Path(r"app.py").resolve().parent   # thư mục chứa app.py

# Dùng dấu "/" (không dùng "\") để chạy được cả Windows lẫn Linux/Streamlit Cloud.
# Đường dẫn tính từ thư mục chứa app.py, KHÔNG phụ thuộc bạn chạy lệnh từ đâu.
MODEL_PATH = "garbage_classifier_4datasets_best.pth"            # file mô hình PyTorch (.pth / .pt), đặt cạnh app.py

# Chỉ cần khi file là state_dict (trọng số). Với file TorchScript hoặc mô hình lưu nguyên
# (torch.save(model)) thì app tự nhận biết và bỏ qua dòng này.
ARCH = "mobilenet_v3_small"         # "mobilenet_v3_large" hoặc "mobilenet_v3_small"

IMG_SIZE = (256, 256)               # (rộng, cao) – đúng bằng kích thước lúc train/test

SUPPORTED_EXT = ("*.pth", "*.pt", "*.pth.tar", "*.ckpt")


def find_model_file(rel_path: str):
    """Tìm file mô hình: thử đường dẫn đã cấu hình, nếu không có thì tìm file .pth/.pt khác trong dự án."""
    p = Path(rel_path.replace("\\", "/"))
    candidates = [p] if p.is_absolute() else [APP_DIR / p, Path.cwd() / p]
    for c in candidates:
        if c.is_file():
            return c
    for pattern in SUPPORTED_EXT:
        found = sorted(APP_DIR.rglob(pattern))
        if found:
            return found[0]
    return None


# Kiểu chuẩn hoá ảnh đầu vào – phải GIỐNG lúc train:
#   "imagenet": (x/255 - mean) / std      -> mặc định của torchvision MobileNetV3
#   "zero_one": x/255                      -> nếu lúc train chỉ dùng ToTensor()
#   "none": giữ nguyên 0..255              -> nếu chuẩn hoá đã nằm trong mô hình
NORMALIZE = "imagenet"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Thứ tự phải TRÙNG với thứ tự lớp lúc huấn luyện (PyTorch ImageFolder: theo bảng chữ cái = train_dataset.classes)
CLASS_NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]

# Thông tin hiển thị cho từng lớp
BIN_COLORS = {
    "Rác tái chế": "#2E86DE",
    "Rác hữu cơ": "#27AE60",
    "Rác thải nguy hại": "#C0392B",
    "Tái sử dụng / quyên góp": "#E67E22",
    "Rác còn lại": "#566573",
}

CLASS_INFO = {
    "battery": {
        "vi": "Pin / ắc quy", "icon": "🔋", "color": "#C0392B",
        "bin": "Rác thải nguy hại",
        "tip": "Không vứt chung rác sinh hoạt. Gom riêng và mang đến điểm thu gom pin hoặc rác thải nguy hại.",
        "avoid": "Không đốt, không đập vỡ hay tháo pin. Pin phồng hoặc rò rỉ cần bọc kín khi mang đi.",
        "decay": "Rất lâu; kim loại bên trong không tự mất đi.",
        "fact": "Tùy loại, pin có thể chứa chì, cadmium, liti... và có thể ngấm vào đất, nước nếu bị chôn lấp.",
    },
    "biological": {
        "vi": "Rác hữu cơ (thực phẩm)", "icon": "🍎", "color": "#27AE60",
        "bin": "Rác hữu cơ",
        "tip": "Tách riêng thức ăn thừa, vỏ rau củ quả. Có thể ủ làm phân compost hoặc dùng làm thức ăn chăn nuôi.",
        "avoid": "Không trộn với nhựa, túi nilon. Đổ bớt nước canh, dầu mỡ thừa trước khi bỏ.",
        "decay": "Vài ngày đến vài tuần (tùy loại và điều kiện).",
        "fact": "Rác hữu cơ chôn lấp lẫn lộn sẽ sinh khí metan, một loại khí nhà kính mạnh.",
    },
    "cardboard": {
        "vi": "Bìa các-tông", "icon": "📦", "color": "#B08968",
        "bin": "Rác tái chế",
        "tip": "Làm phẳng, giữ khô ráo, bỏ băng keo trước khi cho vào thùng tái chế.",
        "avoid": "Bìa dính dầu mỡ (ví dụ hộp pizza bẩn) hoặc bị ướt rất khó tái chế.",
        "decay": "Khoảng vài tháng.",
        "fact": "Tái chế giấy và bìa giúp giảm lượng gỗ phải khai thác để làm giấy mới.",
    },
    "clothes": {
        "vi": "Quần áo / vải", "icon": "👕", "color": "#AF7AC5",
        "bin": "Tái sử dụng / quyên góp",
        "tip": "Đồ còn dùng được hãy quyên góp, bán lại hoặc tái sử dụng làm giẻ lau. Vải rách hỏng mới bỏ vào rác còn lại.",
        "avoid": "Không quyên góp đồ ẩm mốc. Hãy giặt sạch và phơi khô trước.",
        "decay": "Vải tự nhiên: vài tháng đến vài năm. Vải tổng hợp: rất lâu.",
        "fact": "Nhiều loại vải làm từ sợi tổng hợp (như polyester), khi giặt có thể thải ra vi nhựa.",
    },
    "glass": {
        "vi": "Thủy tinh", "icon": "🍾", "color": "#2CB1BC",
        "bin": "Rác tái chế",
        "tip": "Rửa sạch, tháo nắp. Mảnh vỡ cần gói kín, ghi chú để người thu gom không bị đứt tay.",
        "avoid": "Không trộn gương, kính cửa sổ, bóng đèn với chai lọ thủy tinh vì nhiệt độ nóng chảy khác nhau.",
        "decay": "Cực kỳ lâu (hàng nghìn năm trở lên).",
        "fact": "Thủy tinh có thể tái chế nhiều lần mà không giảm chất lượng.",
    },
    "metal": {
        "vi": "Kim loại", "icon": "🥫", "color": "#7F8C8D",
        "bin": "Rác tái chế",
        "tip": "Tráng sơ lon, hộp; ép dẹp nếu có thể để tiết kiệm diện tích.",
        "avoid": "Bình xịt, bình gas mini còn khí hoặc hóa chất cần xử lý riêng, không ép hay đốt.",
        "decay": "Hàng chục đến hàng trăm năm (tùy kim loại và môi trường).",
        "fact": "Tái chế nhôm tiết kiệm phần lớn năng lượng (ước tính khoảng 95%) so với sản xuất từ quặng.",
    },
    "paper": {
        "vi": "Giấy", "icon": "📄", "color": "#F5B041",
        "bin": "Rác tái chế",
        "tip": "Giữ giấy khô và sạch, gom riêng khỏi rác ướt.",
        "avoid": "Giấy dính dầu mỡ, giấy ướt, giấy than hoặc giấy tráng nilon thường không tái chế được.",
        "decay": "Vài tuần đến vài tháng.",
        "fact": "Giấy tái chế được nhiều lần nhưng sợi giấy ngắn dần, nên thường phải trộn thêm bột giấy mới.",
    },
    "plastic": {
        "vi": "Nhựa", "icon": "🧴", "color": "#E74C3C",
        "bin": "Rác tái chế",
        "tip": "Rửa sạch, kiểm tra ký hiệu số tái chế dưới đáy chai hoặc hộp. Tháo nắp nếu có thể.",
        "avoid": "Túi nilon mỏng, nhựa dính thức ăn rất khó tái chế. Không đốt nhựa vì sinh khí độc.",
        "decay": "Hàng trăm năm.",
        "fact": "Nhựa không biến mất mà vỡ thành vi nhựa, có thể đi vào chuỗi thức ăn.",
    },
    "shoes": {
        "vi": "Giày dép", "icon": "👟", "color": "#5D6D7E",
        "bin": "Tái sử dụng / quyên góp",
        "tip": "Giày còn tốt hãy quyên góp hoặc tặng lại. Giày hỏng nặng bỏ vào rác còn lại.",
        "avoid": "Vệ sinh sạch và buộc cặp giày lại với nhau khi quyên góp để không thất lạc.",
        "decay": "Hàng chục đến hàng trăm năm (do đế cao su, nhựa tổng hợp).",
        "fact": "Giày gồm nhiều vật liệu (da, vải, cao su, keo) ghép lại nên rất khó tái chế.",
    },
    "trash": {
        "vi": "Rác còn lại", "icon": "🗑️", "color": "#566573",
        "bin": "Rác còn lại",
        "tip": "Bỏ vào thùng rác thông thường. Hạn chế đồ dùng một lần để giảm lượng rác này.",
        "avoid": "Không bỏ pin, hóa chất hay vật sắc nhọn chưa bọc vào đây.",
        "decay": "Tùy vật liệu.",
        "fact": "Rác còn lại thường bị chôn lấp hoặc đốt, nên giảm rác từ nguồn là cách hiệu quả nhất.",
    },
}
for _v in CLASS_INFO.values():
    _v["bin_color"] = BIN_COLORS[_v["bin"]]

# ============================================================
# CSS TUỲ CHỈNH
# ============================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(135deg, #1D976C 0%, #93F9B9 100%);
        padding: 28px 32px; border-radius: 18px; color: #06371F;
        margin-bottom: 22px;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: 6px 0 0; font-size: 1.05rem; opacity: .85;}
    .result-card {
        border-radius: 16px; padding: 22px 24px; color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,.15);
    }
    .result-card h2 {margin: 0; font-size: 1.9rem;}
    .result-card .sub {opacity: .9; margin-top: 4px;}
    .chip {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(255,255,255,.22); font-size: .9rem; margin-top: 10px;
    }
    .tip-box {
        border-left: 5px solid #1D976C; background: rgba(29,151,108,.10);
        padding: 12px 16px; border-radius: 8px; margin-top: 14px;
    }
    .metric-small {font-size: .85rem; opacity: .7;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HÀM XỬ LÝ
# ============================================================
def _strip_prefix(state_dict: dict) -> dict:
    """Bỏ tiền tố 'module.' (DataParallel) hoặc 'model.' nếu có."""
    out = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def _build_from_state_dict(state_dict: dict):
    """Dựng MobileNetV3 của torchvision rồi nạp trọng số."""
    import torch.nn as nn
    from torchvision import models

    if not hasattr(models, ARCH):
        raise ValueError(f"ARCH='{ARCH}' không hợp lệ (dùng mobilenet_v3_large hoặc mobilenet_v3_small).")
    net = getattr(models, ARCH)(weights=None)

    # Số lớp đầu ra lấy từ chính file trọng số (tránh sai khi bạn đổi số lớp)
    head_keys = [k for k in state_dict if k.startswith("classifier.") and k.endswith(".weight")]
    if not head_keys:
        raise ValueError("Không thấy 'classifier.*.weight' trong file trọng số. Kiểm tra lại ARCH hoặc cách lưu.")
    last_key = sorted(head_keys, key=lambda k: int(k.split(".")[1]))[-1]
    n_out = state_dict[last_key].shape[0]
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_out)

    net.load_state_dict(state_dict, strict=True)
    return net


@st.cache_resource(show_spinner="Đang tải mô hình PyTorch...")
def load_model(path: str):
    """Nạp mô hình PyTorch (CPU). Trả về None nếu không thấy file, hoặc {'error': ...} nếu nạp lỗi."""
    model_file = find_model_file(path)
    if model_file is None:
        return None
    try:
        import torch

        torch.set_num_threads(2)
        kind = None
        net = None

        # 1) TorchScript (torch.jit.save / torch.jit.trace / torch.jit.script)
        try:
            net = torch.jit.load(str(model_file), map_location="cpu")
            kind = "TorchScript"
        except Exception:
            pass

        # 2) torch.save(...): state_dict hoặc mô hình lưu nguyên
        if net is None:
            try:
                obj = torch.load(str(model_file), map_location="cpu", weights_only=True)
            except Exception:
                # Mô hình lưu nguyên cần weights_only=False (chỉ dùng với file CỦA BẠN)
                obj = torch.load(str(model_file), map_location="cpu", weights_only=False)

            if isinstance(obj, torch.nn.Module):
                net, kind = obj, "Mô hình đầy đủ"
            elif isinstance(obj, dict):
                sd = obj
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in obj and isinstance(obj[key], dict):
                        sd = obj[key]
                        break
                net = _build_from_state_dict(_strip_prefix(sd))
                kind = f"state_dict ({ARCH})"
            else:
                raise ValueError(f"Không hiểu định dạng file: {type(obj).__name__}")

        net.eval()

        # Kiểm tra chạy thử và số lớp đầu ra
        w, h = IMG_SIZE
        with torch.inference_mode():
            out = net(torch.zeros(1, 3, h, w))
        n_out = int(out.shape[-1])
        if n_out != len(CLASS_NAMES):
            raise ValueError(
                f"Mô hình có {n_out} đầu ra nhưng CLASS_NAMES có {len(CLASS_NAMES)} lớp. "
                "Hãy sửa CLASS_NAMES cho đúng với lúc train."
            )
        return {"net": net, "kind": kind, "file": model_file, "size": IMG_SIZE}
    except Exception as e:  # hiển thị lỗi thật ở sidebar thay vì chạy demo âm thầm
        return {"error": f"{type(e).__name__}: {e}", "file": model_file}


def preprocess(img: Image.Image):
    import torch

    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    if NORMALIZE == "imagenet":
        arr = (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    elif NORMALIZE == "zero_one":
        arr = arr / 255.0

    arr = np.transpose(arr, (2, 0, 1))                       # HWC -> CHW
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)  # [1,3,H,W]


def predict(model, img: Image.Image) -> np.ndarray:
    """Trả về vector xác suất. Nếu chưa có mô hình -> sinh dữ liệu giả (chế độ demo)."""
    if model is None:
        rng = np.random.default_rng(abs(hash(img.tobytes())) % (2**32))
        logits = rng.normal(size=len(CLASS_NAMES)) * 2
        e = np.exp(logits - logits.max())
        return e / e.sum()

    import torch

    with torch.inference_mode():
        out = model["net"](preprocess(img))
    out = out.squeeze(0).float().numpy()

    if not np.isclose(out.sum(), 1.0, atol=1e-3) or (out < 0).any():  # logits -> softmax
        e = np.exp(out - out.max())
        out = e / e.sum()
    return out


def render_result(img: Image.Image, probs: np.ndarray, threshold: float, top_k: int):
    order = np.argsort(probs)[::-1]
    top = order[0]
    name = CLASS_NAMES[top]
    info = CLASS_INFO.get(name, {"vi": name, "icon": "❓", "color": "#444",
                                 "bin": "Không rõ", "bin_color": "#444", "tip": ""})
    conf = float(probs[top])

    col_img, col_res = st.columns([1, 1.15], gap="large")

    with col_img:
        st.image(img, caption="Ảnh đầu vào", use_container_width=True)

    with col_res:
        if conf < threshold:
            st.warning(
                f"⚠️ Độ tin cậy thấp ({conf:.1%}). Hãy thử chụp lại ảnh rõ hơn, "
                f"đặt vật thể ở giữa và nền đơn giản."
            )
        st.markdown(
            f"""
            <div class="result-card" style="background:{info['color']}">
                <div class="sub">Kết quả dự đoán</div>
                <h2>{info['icon']} {info['vi']}</h2>
                <div class="sub">Độ tin cậy: <b>{conf:.1%}</b></div>
                <span class="chip">🚮 {info['bin']}</span>
            </div>
            <div class="tip-box">💡 <b>Mẹo xử lý:</b> {info['tip']}</div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(conf, 1.0))

        with st.expander("📚 Tìm hiểu thêm về loại rác này"):
            st.markdown(f"**🚫 Lưu ý:** {info.get('avoid', '—')}")
            st.markdown(f"**⏳ Thời gian phân hủy (ước tính):** {info.get('decay', '—')}")
            st.markdown(f"**🌱 Bạn có biết?** {info.get('fact', '—')}")

    st.markdown("#### 📊 Xác suất các lớp")
    df = pd.DataFrame(
        {
            "Loại rác": [f"{CLASS_INFO[CLASS_NAMES[i]]['icon']} {CLASS_INFO[CLASS_NAMES[i]]['vi']}"
                         for i in order[:top_k]],
            "Xác suất": [float(probs[i]) for i in order[:top_k]],
        }
    ).set_index("Loại rác")
    st.bar_chart(df, horizontal=True, height=60 + 42 * top_k)

    return name, conf


def add_history(name: str, conf: float, img: Image.Image):
    thumb = img.copy()
    thumb.thumbnail((120, 120))
    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": name,
            "conf": conf,
            "thumb": thumb,
        },
    )
    st.session_state.history = st.session_state.history[:30]


# ============================================================
# TRẠNG THÁI
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

_loaded = load_model(MODEL_PATH)
load_error = _loaded.get("error") if isinstance(_loaded, dict) else None
model = None if load_error else _loaded

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("♻️ Cài đặt")
    threshold = st.slider("Ngưỡng tin cậy tối thiểu", 0.0, 1.0, 0.6, 0.05,
                          help="Dưới ngưỡng này ứng dụng sẽ hiện cảnh báo.")
    top_k = st.slider("Số lớp hiển thị trong biểu đồ", 2, len(CLASS_NAMES), 4)

    st.divider()
    if load_error:
        st.error("Tìm thấy file mô hình nhưng **nạp bị lỗi** → đang chạy **chế độ DEMO** (kết quả ngẫu nhiên).")
        st.code(load_error)
        st.caption("Gợi ý: kiểm tra `ARCH` (mobilenet_v3_large / small), `CLASS_NAMES` và cách lưu file mô hình.")
    elif model is None:
        st.error(
            f"Không tìm thấy `{MODEL_PATH}`.\n\n"
            f"Đã tìm trong: `{APP_DIR}`\n\n"
            "Đang chạy **chế độ DEMO** (kết quả ngẫu nhiên)."
        )
        with st.expander("File trong thư mục app"):
            st.code("\n".join(sorted(str(p.relative_to(APP_DIR)) for p in APP_DIR.rglob("*")
                                     if p.is_file() and ".git" not in p.parts)[:50]) or "(trống)")
    else:
        st.success("Mô hình PyTorch đã sẵn sàng ✅")
        _mf = model["file"]
        st.caption(f"File: `{_mf.relative_to(APP_DIR) if APP_DIR in _mf.parents else _mf}`")
        st.caption(
            f"Kiểu: {model['kind']} • Đầu vào: {model['size'][0]}×{model['size'][1]} • "
            f"chuẩn hoá: {NORMALIZE} • CPU"
        )

    st.divider()
    st.subheader("Các lớp nhận diện")
    for c in CLASS_NAMES:
        i = CLASS_INFO[c]
        st.markdown(f"{i['icon']} **{i['vi']}** — _{i['bin']}_")

    st.divider()
    if st.button("🧹 Xoá lịch sử", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>♻️ Phân loại rác thải bằng hình ảnh</h1>
        <p>Tải ảnh hoặc chụp trực tiếp — mô hình học sâu sẽ cho biết loại rác và cách bỏ đúng thùng.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_camera, tab_history, tab_about = st.tabs(
    ["📁 Tải ảnh lên", "📷 Chụp ảnh", "🕘 Lịch sử", "ℹ️ Giới thiệu"]
)

# ---------- Tab tải ảnh ----------
with tab_upload:
    files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if files:
        for f in files:
            img = Image.open(f)
            st.markdown(f"##### 🖼️ {f.name}")
            with st.spinner("Đang phân tích..."):
                t0 = time.time()
                probs = predict(model, img)
                dt = time.time() - t0
            name, conf = render_result(img, probs, threshold, top_k)
            st.caption(f"⏱️ Thời gian suy luận: {dt*1000:.0f} ms")
            add_history(name, conf, img)
            st.divider()
    else:
        st.info("👆 Hãy tải ảnh rác thải lên để bắt đầu (Lưu ý: ảnh phải rõ nét).")

# ---------- Tab camera ----------
with tab_camera:
    shot = st.camera_input("Chụp ảnh vật thể cần phân loại")
    if shot:
        img = Image.open(shot)
        with st.spinner("Đang phân tích..."):
            probs = predict(model, img)
        name, conf = render_result(img, probs, threshold, top_k)
        add_history(name, conf, img)

# ---------- Tab lịch sử ----------
with tab_history:
    hist = st.session_state.history
    if not hist:
        st.info("Chưa có lượt phân loại nào.")
    else:
        counts = pd.Series([h["name"] for h in hist]).value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng lượt phân loại", len(hist))
        top_name = counts.index[0]
        c2.metric("Loại phổ biến nhất", CLASS_INFO[top_name]["vi"])
        c3.metric("Độ tin cậy trung bình", f"{np.mean([h['conf'] for h in hist]):.1%}")

        st.markdown("#### Thống kê theo loại")
        st.bar_chart(counts.rename(index=lambda x: CLASS_INFO[x]["vi"]))

        st.markdown("#### Chi tiết")
        for h in hist:
            a, b = st.columns([1, 5])
            a.image(h["thumb"])
            info = CLASS_INFO[h["name"]]
            b.markdown(
                f"**{info['icon']} {info['vi']}** — {h['conf']:.1%}  \n"
                f"<span class='metric-small'>{h['time']} • {info['bin']}</span>",
                unsafe_allow_html=True,
            )

# ---------- Tab giới thiệu ----------
with tab_about:
    st.markdown(
        """
        ### Về dự án
        Ứng dụng sử dụng mạng nơ-ron tích chập (CNN) để nhận diện **10 nhóm rác thải**
        từ ảnh chụp, giúp người dùng phân loại đúng thùng rác và góp phần bảo vệ môi trường.

        ### Cách sử dụng
        1. Vào tab **Tải ảnh lên** hoặc **Chụp ảnh**.
        2. Xem kết quả, độ tin cậy và gợi ý xử lý.
        3. Theo dõi thống kê ở tab **Lịch sử**.

        ### Mẹo để kết quả chính xác
        - Chụp một vật thể mỗi ảnh, đặt ở giữa khung hình.
        - Ánh sáng đủ, nền đơn giản.
        - Tránh ảnh mờ hoặc bị che khuất.
        """
    )
    st.caption("Làm bằng Python • Streamlit • PyTorch • Github")
