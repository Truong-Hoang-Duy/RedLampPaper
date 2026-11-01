"""
RedLamp Interactive Studio - Demo Đồ Án Cuối Kỳ CS2207 (Data Mining & Applications)
Bài báo: Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (KDD '25)

Chạy ứng dụng:
    streamlit run demo/app.py
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Thêm thư mục gốc của project vào sys.path để import trực tiếp models, loaders, main
DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import utils
from models.meta import ConvAEC
from loaders.dataset import Entity, Dataset
from loaders.loader_aug import Loader_aug
from main import convolve_minmax_score, mse, label_score_selected_feature
from demo.generate_sample_data import ensure_sample_data, ANOMALY_TYPES


st.set_page_config(
    page_title="RedLamp Interactive Studio | KDD '25 Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_COLORS = {
    'normal': '#2ecc71',
    'spike': '#e74c3c',
    'flip': '#e67e22',
    'speedup': '#f1c40f',
    'noise': '#1abc9c',
    'cutoff': '#9b59b6',
    'average': '#3498db',
    'scale': '#e84393',
    'wander': '#00cec9',
    'contextual': '#6c5ce7',
    'upsidedown': '#fd79a8',
    'mixture': '#a29bfe',
}

CLASS_DESCRIPTIONS_VI = {
    'normal': 'Chuỗi gốc không biến đổi (k = 1)',
    'spike': 'Chèn gai nhọn đột ngột tại 1 thời điểm (Point outlier)',
    'flip': 'Lật ngược thứ tự thời gian trong đoạn [st, ed]',
    'speedup': 'Tăng gấp đôi (2x) hoặc giãn một nửa (1/2x) tần số',
    'noise': 'Cộng nhiễu trắng Gaussian N(0, 0.1I) vào đoạn [st, ed]',
    'cutoff': 'Cắt phẳng tín hiệu về một giá trị hằng số ngẫu nhiên',
    'average': 'Làm trơn tín hiệu bằng trung bình trượt (Moving Average)',
    'scale': 'Nhân biên độ đoạn [st, ed] với hệ số ngẫu nhiên',
    'wander': 'Thêm xu hướng tuyến tính (trend) và dịch chuyển nền (baseline shift)',
    'contextual': 'Kết hợp co giãn biên độ và dịch chuyển đường cơ sở',
    'upsidedown': 'Lật ngược cực tính tín hiệu qua giá trị trung bình của đoạn',
    'mixture': 'Tráo đổi đoạn [st, ed] với một đoạn khác lấy từ vị trí ngẫu nhiên',
}


@st.cache_resource
def init_demo_samples():
    ensure_sample_data()
    return True


def discover_experiment_runs():
    """
    Tự động quét tất cả các thư mục chứa kết quả chạy thực nghiệm (input.npy, pred.npy, ...)
    trong ./result (dữ liệu thật tải từ Google Drive/main.py) và ./demo/sample_results (dữ liệu mẫu).
    """
    runs = {}
    search_roots = [
        ("Dữ liệu thực nghiệm (./result)", ROOT_DIR / "result"),
        ("Dữ liệu Demo mẫu (./demo/sample_results)", DEMO_DIR / "sample_results"),
    ]
    for source_label, root in search_roots:
        if not root.exists():
            continue
        for input_file in sorted(root.rglob("input.npy")):
            test_dir = input_file.parent
            rel_path = test_dir.relative_to(root)
            parts = rel_path.parts
            # Đặt tên hiển thị dễ đọc
            display_name = f"[{source_label.split('(')[0].strip()}] " + " / ".join(parts[:-1] if parts[-1] == "test_all" else parts)
            runs[display_name] = test_dir
    return runs


def load_run_bundle(test_dir: Path):
    """Đọc toàn bộ file .npy và metadata từ một thư mục test_all."""
    bundle = {
        "test_dir": test_dir,
        "model_dir": test_dir.parent,
        "input": np.load(test_dir / "input.npy"),
        "pred": np.load(test_dir / "pred.npy"),
        "pred_label": np.load(test_dir / "pred_label.npy"),
        "enc": np.load(test_dir / "enc.npy"),
        "anomaly_score": np.load(test_dir / "anomaly_score.npy"),
    }
    if (test_dir / "anomaly_mask.npy").exists():
        bundle["anomaly_mask"] = np.load(test_dir / "anomaly_mask.npy")
    if (test_dir / "gt_labels.npy").exists():
        bundle["gt_labels"] = np.load(test_dir / "gt_labels.npy")
    if (test_dir / "meta.json").exists():
        with open(test_dir / "meta.json", "r", encoding="utf-8") as f:
            bundle["meta"] = json.load(f)
    else:
        bundle["meta"] = {}
    if (test_dir / "train_aug_enc.npy").exists() and (test_dir / "train_aug_cls.npy").exists():
        bundle["train_aug_enc"] = np.load(test_dir / "train_aug_enc.npy")
        bundle["train_aug_cls"] = np.load(test_dir / "train_aug_cls.npy")
    return bundle


def compute_redlamp_scores(inputs, preds, pred_labels, faa_threshold=0.05, mse_weight=0.5, smooth_w=50):
    """
    Tính toán chi tiết các thành phần điểm bất thường theo Mục 3.6 của bài báo RedLamp:
    - s_MSE(t): Reconstruction error
    - s_CE(t): Adjusted anomaly-class score sau khi qua bộ lọc FAA (Frequent Anomaly Adjustment)
    - s(t): Tổng trọng số sau chuẩn hóa MinMax
    """
    B, W, D = inputs.shape
    w = max(1, int(smooth_w))

    flat_in = inputs.reshape(B, -1)
    flat_pred = preds.reshape(B, -1)
    raw_mse = mse(flat_in, flat_pred)
    norm_mse = convolve_minmax_score(raw_mse.reshape(-1, 1), w=w).squeeze()

    # Tần suất dự đoán trung bình của từng lớp trên toàn tập test (Eq. 6)
    mean_class_probs = np.mean(pred_labels, axis=0)
    # Lớp 0 ('normal') luôn bị loại khỏi tổng điểm bất thường; các lớp > faa_threshold là pseudo-normal
    suppressed_indices = np.where(mean_class_probs > faa_threshold)[0]
    if 0 not in suppressed_indices:
        suppressed_indices = np.insert(suppressed_indices, 0, 0)

    raw_ce = label_score_selected_feature(pred_labels, axis=[0])
    norm_raw_ce = convolve_minmax_score(raw_ce.reshape(-1, 1), w=w).squeeze()

    faa_ce = label_score_selected_feature(pred_labels, axis=suppressed_indices)
    norm_faa_ce = convolve_minmax_score(faa_ce.reshape(-1, 1), w=w).squeeze()

    total_score = mse_weight * norm_mse + (1.0 - mse_weight) * norm_faa_ce
    total_score_no_faa = mse_weight * norm_mse + (1.0 - mse_weight) * norm_raw_ce

    return {
        "norm_mse": norm_mse,
        "norm_raw_ce": norm_raw_ce,
        "norm_faa_ce": norm_faa_ce,
        "total_score": total_score,
        "total_score_no_faa": total_score_no_faa,
        "mean_class_probs": mean_class_probs,
        "suppressed_indices": suppressed_indices,
    }


def compute_point_adjusted_f1(gt, score, threshold):
    """Tính F1 thông thường và Point-Adjusted F1 trên chuỗi test."""
    pred = (score >= threshold).astype(int)
    raw_f1 = f1_score(gt, pred, zero_division=0)

    pa_pred = pred.copy()
    # Tìm các đoạn bất thường liên tiếp trong ground truth
    diff = np.diff(np.concatenate(([0], gt, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if np.any(pa_pred[s:e] == 1):
            pa_pred[s:e] = 1
    pa_f1 = f1_score(gt, pa_pred, zero_division=0)
    return raw_f1, pa_f1


# Khởi tạo dữ liệu mẫu nếu chưa có
init_demo_samples()

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("## 🚨 RedLamp Studio")
    st.caption("**KDD '25 Final Project Demo** — CS2207 Data Mining & Applications")

    runs_map = discover_experiment_runs()
    selected_run_label = st.selectbox(
        "📂 Chọn bộ kết quả thực nghiệm:",
        options=list(runs_map.keys()),
        index=0,
    )
    bundle = load_run_bundle(runs_map[selected_run_label])

    st.markdown("---")
    st.markdown("### ⚙️ Tham số Suy luận (Mục 3.6)")
    faa_delta = st.slider(
        "Ngưỡng FAA (δ - lọc lớp giả bình thường):",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="Nếu xác suất trung bình của một lớp bất thường trên toàn chuỗi test > δ, lớp đó được coi là Pseudo-Normal và bị loại bỏ (Eq. 6).",
    )
    mse_weight = st.slider(
        "Trọng số Reconstruction vs. Classifier:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Trong bài báo (Eq. 8), điểm tổng hợp lấy trung bình cộng (0.5 * s_MSE + 0.5 * s_CE).",
    )
    smooth_w = st.slider(
        "Cửa sổ làm mượt điểm (Moving Average w):",
        min_value=1,
        max_value=100,
        value=50,
        step=5,
    )

    st.markdown("---")
    st.info(
        "💡 **Tích hợp Google Drive:** Bạn có thể chép bất kỳ thư mục kết quả nào từ Google Drive vào `./result/` — ứng dụng sẽ tự động nhận diện ngay lập tức!"
    )

# ======================== HEADER ========================
st.title("🚨 RedLamp: Robust & Explainable Time Series Anomaly Detector")
st.markdown(
    "**Bài báo gốc (KDD '25):** *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies* "
    "— Kohei Obata, Yasuko Matsubara, Yasushi Sakurai (University of Osaka)."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 Tab 1: Đa dạng hóa Giả Bất thường & Nhãn mềm (Mục 3.3–3.4)",
    "📊 Tab 2: Giải phẫu Điểm Bất thường & Bộ lọc FAA (Mục 3.5–3.6)",
    "🧭 Tab 3: Không gian Tiềm ẩn Giải thích được - t-SNE (Mục 4.6)",
    "⚡ Tab 4: Chẩn đoán Trực tiếp - Live Anomaly Injection",
])

# ==============================================================================
# TAB 1: DATA AUGMENTATION & LABEL REFURBISHMENT PLAYGROUND
# ==============================================================================
with tab1:
    st.subheader("1. Sinh Đa Lớp Giả Bất Thường (11 Time-Series Data Augmentations)")
    st.markdown(
        "Khác với hướng tiếp cận **Normality Assumption** (dễ học vẹt cả nhiễu trong tập train) hay **Binary Anomaly Assumption** "
        "(gặp vấn đề *Diversity Gap* và *False Anomaly*), **RedLamp** sinh **11 lớp giả bất thường chuyên biệt cho chuỗi thời gian** "
        "kèm mặt nạ vị trí `anomaly_mask` $\\mathbf{W}^{(t,k)}$."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1.2, 1.2, 1.6])
    with col_cfg1:
        aug_seed = st.number_input("Random Seed tạo biến đổi:", min_value=0, max_value=999, value=42, step=1)
    with col_cfg2:
        focus_aug = st.selectbox(
            "🔍 Chọn 1 phép biến đổi để soi chi tiết:",
            options=ANOMALY_TYPES[1:],
            index=0,
            format_func=lambda x: f"{x.upper()} — {CLASS_DESCRIPTIONS_VI[x]}",
        )
    with col_cfg3:
        col_a, col_b = st.columns(2)
        with col_a:
            alpha_val = st.slider("Xác suất nhầm về Normal (α):", 0.0, 0.30, 0.10, 0.01)
        with col_b:
            beta_val = st.slider("Xác suất nhầm giữa các lớp (β):", 0.0, 0.04, 0.01, 0.002)

    # Lấy 1 chuỗi mẫu từ tập dữ liệu hiện tại để áp dụng Loader_aug thật
    raw_series = bundle["input"][:, -1, 0]
    if len(raw_series) < 300:
        sample_seq = np.tile(raw_series, 3)[:300].reshape(1, -1)
    else:
        sample_seq = raw_series[:300].reshape(1, -1)

    dummy_entity = Entity(Y=sample_seq, name="demo_aug", verbose=False)
    dummy_dataset = Dataset(entities=[dummy_entity], name="demo_aug", verbose=False)
    aug_helper = Loader_aug(
        dataset=dummy_dataset,
        batch_size=16,
        window_size=100,
        window_step=50,
        anomaly_types=['normal'],
        anomaly_types_for_dict=ANOMALY_TYPES,
        min_range=18,
        min_features=1,
        max_features=1,
        fast_sampling=False,
        shuffle=False,
        verbose=False,
    )

    # Vẽ chi tiết phép biến đổi được chọn + Soft Label (Backward Correction)
    np.random.seed(int(aug_seed))
    y_aug_detail, z_norm_detail, mask_detail = aug_helper.select_anomalies(focus_aug, sample_seq, 50, 150)
    y_aug_1d = y_aug_detail[0]
    z_norm_1d = z_norm_detail[0]
    mask_1d = mask_detail[0]  # Lưu ý trong code: 1 là vùng bình thường, 0 là vùng chèn anomaly!

    c_left, c_right = st.columns([1.35, 1.0])
    with c_left:
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Scatter(
            y=z_norm_1d, mode='lines', name='Chuỗi gốc bình thường X^(t)',
            line=dict(color='#7f8c8d', width=2, dash='dot')
        ))
        fig_detail.add_trace(go.Scatter(
            y=y_aug_1d, mode='lines', name=f'Sau khi chèn {focus_aug.upper()}',
            line=dict(color=CLASS_COLORS[focus_aug], width=2.8)
        ))
        # Tô vùng anomaly_mask == 0 (nơi có bất thường được chèn)
        anom_pts = np.where(mask_1d == 0)[0]
        if len(anom_pts) > 0:
            fig_detail.add_vrect(
                x0=int(anom_pts.min()), x1=int(anom_pts.max()) + 1,
                fillcolor="rgba(231, 76, 60, 0.22)", line_width=0,
                annotation_text="Vùng Masked (không tính L_MSE)", annotation_position="top left"
            )
        fig_detail.update_layout(
            title=f"So sánh Tín hiệu gốc vs. Giả bất thường [{focus_aug.upper()}] (Cửa sổ L=100)",
            xaxis_title="Timestep trong cửa sổ",
            yaxis_title="Giá trị chuẩn hóa",
            height=340,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_detail, use_container_width=True)

    with c_right:
        K = len(ANOMALY_TYPES)
        k_idx = ANOMALY_TYPES.index(focus_aug)
        one_hot = np.zeros(K)
        one_hot[k_idx] = 1.0
        # Công thức (1) trong bài báo & models/meta.py dòng 52-53
        soft_label = one_hot * (1.0 - alpha_val - K * beta_val) + beta_val
        soft_label[0] += alpha_val

        fig_lbl = go.Figure()
        fig_lbl.add_trace(go.Bar(
            x=ANOMALY_TYPES, y=one_hot, name="Nhãn cứng (One-hot)",
            marker_color="rgba(149, 165, 166, 0.55)"
        ))
        fig_lbl.add_trace(go.Bar(
            x=ANOMALY_TYPES, y=soft_label, name="Nhãn mềm (Backward Correction ỹ)",
            marker_color=[CLASS_COLORS[c] for c in ANOMALY_TYPES]
        ))
        fig_lbl.update_layout(
            title=f"Label Refurbishment (Eq. 1): α={alpha_val:.2f}, β={beta_val:.3f}",
            yaxis_title="Xác suất nhãn mục tiêu",
            barmode="group",
            height=340,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_lbl, use_container_width=True)

    with st.expander("🖼️ Xem toàn cảnh 12 Lớp Biến Đổi Chuỗi Thời Gian (Tái hiện trực tiếp Figure 2a trong bài báo)", expanded=True):
        fig_grid = make_subplots(
            rows=3, cols=4,
            subplot_titles=[f"({i+1}) {t.capitalize()}" for i, t in enumerate(ANOMALY_TYPES)],
            vertical_spacing=0.12, horizontal_spacing=0.05
        )
        for idx, atype in enumerate(ANOMALY_TYPES):
            r = idx // 4 + 1
            c = idx % 4 + 1
            np.random.seed(int(aug_seed) + idx * 7)
            y_a, z_n, m_a = aug_helper.select_anomalies(atype, sample_seq, 50, 150)
            fig_grid.add_trace(
                go.Scatter(y=z_n[0], mode='lines', line=dict(color='#95a5a6', width=1.2, dash='dot'), showlegend=False),
                row=r, col=c
            )
            fig_grid.add_trace(
                go.Scatter(y=y_a[0], mode='lines', line=dict(color=CLASS_COLORS[atype], width=2), showlegend=False),
                row=r, col=c
            )
            apts = np.where(m_a[0] == 0)[0]
            if len(apts) > 0:
                fig_grid.add_vrect(
                    x0=int(apts.min()), x1=int(apts.max()) + 1,
                    fillcolor="rgba(231, 76, 60, 0.18)", line_width=0,
                    row=r, col=c
                )
        fig_grid.update_layout(height=580, margin=dict(l=15, r=15, t=40, b=15))
        st.plotly_chart(fig_grid, use_container_width=True)

# ==============================================================================
# TAB 2: ANOMALY SCORING & FAA INSPECTOR
# ==============================================================================
with tab2:
    st.subheader("2. Giải phẫu Điểm Bất thường & Cơ chế Lọc Lớp Giả Bình Thường (FAA)")
    if bundle["meta"].get("description"):
        st.caption(f"📌 **Mô tả bộ dữ liệu đang chọn:** {bundle['meta']['description']}")

    inputs = bundle["input"]
    preds = bundle["pred"]
    pred_labels = bundle["pred_label"]
    B, W, D = inputs.shape

    feature_idx = 0
    if D > 1:
        feature_idx = st.selectbox("Chọn chiều đặc trưng (Feature channel) để hiển thị:", list(range(D)), index=0)

    scores_dict = compute_redlamp_scores(
        inputs, preds, pred_labels,
        faa_threshold=faa_delta,
        mse_weight=mse_weight,
        smooth_w=smooth_w,
    )

    # Căn chỉnh độ dài chuỗi thời gian giống main.py (dòng 476-478)
    seq_in = np.concatenate([np.zeros(W - 1), inputs[:, -1, feature_idx]])
    seq_pred = np.concatenate([np.zeros(W - 1), preds[:, -1, feature_idx]])
    score_total = np.concatenate([np.zeros(W - 1), scores_dict["total_score"]])
    score_no_faa = np.concatenate([np.zeros(W - 1), scores_dict["total_score_no_faa"]])
    score_mse = np.concatenate([np.zeros(W - 1), scores_dict["norm_mse"]])
    score_ce = np.concatenate([np.zeros(W - 1), scores_dict["norm_faa_ce"]])

    total_len = len(seq_in)
    if "gt_labels" in bundle and len(bundle["gt_labels"]) == total_len:
        gt_vec = bundle["gt_labels"]
    else:
        gt_vec = np.zeros(total_len, dtype=int)
        a_st = bundle["meta"].get("anomaly_start_in_test", 0)
        a_ed = bundle["meta"].get("anomaly_end_in_test", 0)
        if a_ed > a_st:
            gt_vec[a_st:a_ed] = 1

    # Tính các chỉ số đánh giá nếu có nhãn Ground Truth
    valid_slice = slice(W - 1, total_len)
    if np.any(gt_vec[valid_slice] == 1) and np.any(gt_vec[valid_slice] == 0):
        auc_roc = roc_auc_score(gt_vec[valid_slice], score_total[valid_slice])
        auc_pr = average_precision_score(gt_vec[valid_slice], score_total[valid_slice])
        auc_pr_nofaa = average_precision_score(gt_vec[valid_slice], score_no_faa[valid_slice])
        best_th = np.percentile(score_total[valid_slice], 95)
        raw_f1, pa_f1 = compute_point_adjusted_f1(gt_vec[valid_slice], score_total[valid_slice], best_th)
        top1_hit = "✅ Chính xác (Hit)" if gt_vec[np.argmax(score_total)] == 1 else "❌ Lệch (Miss)"

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC-ROC (Có FAA)", f"{auc_roc:.4f}")
        m2.metric("AUC-PR (Có FAA)", f"{auc_pr:.4f}", delta=f"{auc_pr - auc_pr_nofaa:+.4f} so với tắt FAA")
        m3.metric("Point-Adjusted F1", f"{pa_f1:.4f}")
        m4.metric("UCR Top-1 Accuracy", top1_hit)
        suppressed_names = [ANOMALY_TYPES[i] for i in scores_dict["suppressed_indices"] if i != 0]
        m5.metric("Lớp Pseudo-Normal bị FAA lọc", ", ".join(suppressed_names) if suppressed_names else "Không có")

    # Biểu đồ đồng bộ 3 tầng: (1) Tín hiệu & Tái tạo, (2) Điểm thành phần MSE vs CE, (3) Điểm tổng hợp Có/Không FAA
    fig_sync = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "1. Chuỗi Thời Gian Đầu Vào X(t) vs. Tái tạo từ Autoencoder X̂(t)",
            "2. Hai thành phần Điểm Bất Thường: Reconstruction Score s_MSE(t) vs. FAA Classifier Score s_CE(t)",
            "3. Điểm Bất Thường Tổng Hợp RedLamp s(t): Khi BẬT FAA vs. Khi TẮT FAA (w/o FAA)",
        ),
        row_heights=[0.36, 0.32, 0.32],
    )

    fig_sync.add_trace(go.Scatter(y=seq_in, name="Input X(t)", line=dict(color="#2c3e50", width=1.6)), row=1, col=1)
    fig_sync.add_trace(go.Scatter(y=seq_pred, name="Reconstruction X̂(t)", line=dict(color="#e67e22", width=1.5, dash="dot")), row=1, col=1)

    fig_sync.add_trace(go.Scatter(y=score_mse, name="s_MSE(t) (Reconstruction)", line=dict(color="#2980b9", width=1.7)), row=2, col=1)
    fig_sync.add_trace(go.Scatter(y=score_ce, name="s_CE(t) (Multiclass + FAA)", line=dict(color="#8e44ad", width=1.7)), row=2, col=1)

    fig_sync.add_trace(go.Scatter(y=score_no_faa, name="Tắt FAA (w/o FAA)", line=dict(color="#95a5a6", width=1.5, dash="dash")), row=3, col=1)
    fig_sync.add_trace(go.Scatter(y=score_total, name="RedLamp s(t) (Có FAA)", line=dict(color="#c0392b", width=2.4)), row=3, col=1)

    # Tô nền đỏ các khoảng Ground Truth Anomaly
    diff_gt = np.diff(np.concatenate(([0], gt_vec, [0])))
    gt_starts = np.where(diff_gt == 1)[0]
    gt_ends = np.where(diff_gt == -1)[0]
    for s_idx, e_idx in zip(gt_starts, gt_ends):
        for r_idx in (1, 2, 3):
            fig_sync.add_vrect(
                x0=int(s_idx), x1=int(e_idx),
                fillcolor="rgba(231, 76, 60, 0.22)", line_width=0,
                row=r_idx, col=1
            )

    fig_sync.update_layout(
        height=680,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sync, use_container_width=True)

    # Biểu đồ kiểm tra ngưỡng FAA (Eq. 6)
    col_faa1, col_faa2 = st.columns([1.1, 1.3])
    with col_faa1:
        mean_probs = scores_dict["mean_class_probs"][1:]  # Bỏ lớp normal ở vị trí 0
        anom_names = ANOMALY_TYPES[1:]
        bar_colors = [
            "#95a5a6" if p > faa_delta else CLASS_COLORS[c]
            for p, c in zip(mean_probs, anom_names)
        ]
        fig_faa_bar = go.Figure()
        fig_faa_bar.add_trace(go.Bar(
            x=anom_names, y=mean_probs,
            marker_color=bar_colors,
            text=[f"{p:.3f}" for p in mean_probs],
            textposition="outside",
        ))
        fig_faa_bar.add_hline(
            y=faa_delta, line_dash="dash", line_color="#e74c3c",
            annotation_text=f"Ngưỡng FAA δ = {faa_delta:.2f} (Lớp vượt ngưỡng bị coi là Pseudo-Normal)",
            annotation_position="top right"
        )
        fig_faa_bar.update_layout(
            title="Tần suất dự đoán trung bình của 11 lớp trên tập Test (Eq. 6)",
            yaxis_title="Xác suất trung bình",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_faa_bar, use_container_width=True)

    with col_faa2:
        fig_prob_area = go.Figure()
        for c_idx in range(1, len(ANOMALY_TYPES)):
            cname = ANOMALY_TYPES[c_idx]
            is_suppressed = c_idx in scores_dict["suppressed_indices"]
            fig_prob_area.add_trace(go.Scatter(
                y=pred_labels[:, c_idx],
                mode='lines',
                name=f"{cname} {'(FAA=0)' if is_suppressed else ''}",
                line=dict(color=CLASS_COLORS[cname], width=1.5 if not is_suppressed else 1.0, dash='dot' if is_suppressed else 'solid'),
                opacity=0.35 if is_suppressed else 0.9,
            ))
        fig_prob_area.update_layout(
            title="Diễn biến Xác suất dự đoán 11 lớp giả bất thường theo từng cửa sổ thời gian",
            xaxis_title="Chỉ số cửa sổ trượt (Window index t)",
            yaxis_title="Xác suất ŷ_i^(t)",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_prob_area, use_container_width=True)

# ==============================================================================
# TAB 3: EXPLAINABLE LATENT SPACE (t-SNE - FIGURE 6 REPRODUCTION)
# ==============================================================================
with tab3:
    st.subheader("3. Không gian Tiềm ẩn Giải thích được (Explainable Latent Space — Tái hiện Figure 6)")
    st.markdown(
        "Vì nhánh Classifier được huấn luyện để phân tách **12 lớp biến đổi** từ cùng vector biểu diễn `enc.npy` (128 chiều), "
        "không gian tiềm ẩn của **RedLamp** tự nhiên mang tính giải thích: **Các điểm bất thường thực tế trên tập Test sẽ rơi vào gần cụm của lớp giả bất thường có cùng bản chất vật lý!**"
    )

    col_tsne_left, col_tsne_right = st.columns([1.45, 1.0])

    enc_raw = bundle["enc"].reshape(bundle["enc"].shape[0], -1)
    win_gt = gt_vec[W - 1:]
    if len(win_gt) != len(enc_raw):
        win_gt = np.zeros(len(enc_raw), dtype=int)

    with col_tsne_right:
        proj_method = st.radio(
            "Phương pháp giảm chiều từ 128D → 2D:",
            options=["t-SNE (Chuẩn bài báo Figure 6)", "PCA (Tức thì)"],
            horizontal=True,
        )
        # Xác định tổ hợp chữ ký bất thường (Anomaly Signature) trong vùng bất thường
        anom_win_idx = np.where(win_gt == 1)[0]
        if len(anom_win_idx) == 0:
            anom_win_idx = np.argsort(scores_dict["total_score"])[-25:]

        anom_probs_mean = np.mean(pred_labels[anom_win_idx, 1:], axis=0)
        top3_idx = np.argsort(anom_probs_mean)[::-1][:3]
        top1_class = ANOMALY_TYPES[1 + top3_idx[0]]

        st.success(
            f"🔍 **Chẩn đoán giải thích (Explainability):** Vùng bất thường trong chuỗi có đặc trưng gần nhất với lớp "
            f"**`{top1_class.upper()}`** ({CLASS_DESCRIPTIONS_VI[top1_class]})."
        )

        fig_radar = go.Figure(go.Bar(
            x=anom_probs_mean,
            y=[c.capitalize() for c in ANOMALY_TYPES[1:]],
            orientation='h',
            marker_color=[CLASS_COLORS[c] for c in ANOMALY_TYPES[1:]],
            text=[f"{v*100:.1f}%" for v in anom_probs_mean],
            textposition="outside",
        ))
        fig_radar.update_layout(
            title="Phân bố Xác suất Giải thích Loại Bất Thường tại Vùng Đỉnh",
            xaxis_title="Xác suất trung bình của Classifier",
            height=380,
            margin=dict(l=20, r=40, t=45, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_tsne_left:
        # Lấy mẫu các điểm Normal (Test) và Anomaly (Test) + 12 lớp Augmentation (Train)
        norm_win_idx = np.where(win_gt == 0)[0]
        rng_sub = np.random.RandomState(0)
        if len(norm_win_idx) > 80:
            norm_win_idx = rng_sub.choice(norm_win_idx, size=80, replace=False)
        if len(anom_win_idx) > 60:
            anom_sample_idx = rng_sub.choice(anom_win_idx, size=60, replace=False)
        else:
            anom_sample_idx = anom_win_idx

        points_list = []
        tags_list = []

        if "train_aug_enc" in bundle and "train_aug_cls" in bundle:
            points_list.append(bundle["train_aug_enc"])
            for c_id in bundle["train_aug_cls"]:
                tags_list.append(ANOMALY_TYPES[int(c_id)])

        points_list.append(enc_raw[norm_win_idx])
        tags_list.extend(["Normal (Test)"] * len(norm_win_idx))

        points_list.append(enc_raw[anom_sample_idx])
        tags_list.extend(["Anomaly (Test)"] * len(anom_sample_idx))

        all_pts = np.vstack(points_list)
        if proj_method.startswith("t-SNE"):
            perp = min(25, max(5, len(all_pts) // 6))
            coords_2d = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto").fit_transform(all_pts)
        else:
            coords_2d = PCA(n_components=2, random_state=42).fit_transform(all_pts)

        fig_tsne = go.Figure()
        tags_arr = np.array(tags_list)

        # Vẽ 12 lớp Augmentation trên tập Train trước (hình tròn mờ như Figure 6(i))
        for atype in ANOMALY_TYPES:
            mask_t = tags_arr == atype
            if np.any(mask_t):
                fig_tsne.add_trace(go.Scatter(
                    x=coords_2d[mask_t, 0], y=coords_2d[mask_t, 1],
                    mode='markers',
                    name=f"Train: {atype.capitalize()}",
                    marker=dict(size=7, color=CLASS_COLORS[atype], opacity=0.65, symbol='circle'),
                ))

        # Vẽ Normal (Test) và Anomaly (Test) nổi bật
        mask_nt = tags_arr == "Normal (Test)"
        if np.any(mask_nt):
            fig_tsne.add_trace(go.Scatter(
                x=coords_2d[mask_nt, 0], y=coords_2d[mask_nt, 1],
                mode='markers',
                name="Normal (Test)",
                marker=dict(size=9, color="#27ae60", symbol="diamond", line=dict(width=1, color="black")),
            ))

        mask_at = tags_arr == "Anomaly (Test)"
        if np.any(mask_at):
            fig_tsne.add_trace(go.Scatter(
                x=coords_2d[mask_at, 0], y=coords_2d[mask_at, 1],
                mode='markers',
                name="🔥 Anomaly (Test)",
                marker=dict(size=12, color="#e74c3c", symbol="x", line=dict(width=2, color="#c0392b")),
            ))

        fig_tsne.update_layout(
            title="Biểu đồ Không gian Tiềm ẩn 2D (Embedding Space — Tương ứng Figure 6(i))",
            xaxis_title="Chiều tiềm ẩn 1",
            yaxis_title="Chiều tiềm ẩn 2",
            height=480,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

# ==============================================================================
# TAB 4: LIVE ANOMALY INJECTION & REAL-TIME INFERENCE
# ==============================================================================
with tab4:
    st.subheader("4. Thử thách Mô hình Trực tiếp (Live Anomaly Injection & Checkpoint Inference)")
    st.markdown(
        "Dành cho phần **Q&A / Phản biện với Giảng viên**: Chọn một cửa sổ tín hiệu bình thường bất kỳ, "
        "**tự tay chèn một loại đột biến** vào vị trí tùy ý, và chạy suy luận trực tiếp qua mô hình `ConvAEC` (`bestmodel.pkl`)."
    )

    c_inj1, c_inj2, c_inj3 = st.columns(3)
    with c_inj1:
        win_idx_pick = st.slider("Chọn cửa sổ cơ sở từ tập Test:", 0, max(0, len(inputs) - 1), 50)
    with c_inj2:
        live_aug_type = st.selectbox(
            "Loại đột biến muốn bơm vào:",
            options=ANOMALY_TYPES,
            index=1,
            format_func=lambda x: f"{x.upper()} — {CLASS_DESCRIPTIONS_VI[x]}",
        )
    with c_inj3:
        live_seed = st.slider("Seed vị trí/cường độ đột biến:", 0, 100, 12)

    base_window = inputs[win_idx_pick].T.copy()  # (D, W)
    # Bơm đột biến bằng Loader_aug thật
    live_entity = Entity(Y=np.tile(base_window, (1, 3)), name="live", verbose=False)
    live_ds = Dataset(entities=[live_entity], name="live", verbose=False)
    live_loader = Loader_aug(
        dataset=live_ds,
        batch_size=4,
        window_size=W,
        window_step=W,
        anomaly_types=['normal'],
        anomaly_types_for_dict=ANOMALY_TYPES,
        min_range=15,
        min_features=1,
        max_features=D,
        fast_sampling=False,
        shuffle=False,
        verbose=False,
    )
    np.random.seed(int(live_seed))
    y_live, z_live, mask_live = live_loader.select_anomalies(live_aug_type, live_entity.Y, W, 2 * W)

    # Load checkpoint bestmodel.pkl nếu tồn tại để chạy forward thật
    ckpt_path = bundle["model_dir"] / "bestmodel.pkl"
    params_path = bundle["model_dir"] / "model_params.json"
    if ckpt_path.exists():
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                p_dict = json.load(f)
            m_params = utils.AttrDict(**p_dict)
        else:
            m_params = utils.AttrDict(
                name='ConvAEC', n_features=D, n_time=W,
                num_filters=[128, 128, 256, 256], embedding_dim=128,
                kernel_size=4, dropout=0.2, normalization='batch',
                stride=2, padding=2, anomaly_types=ANOMALY_TYPES,
                classes=len(ANOMALY_TYPES), classifier_dim=32,
                c_loss_ratio=0.1, apply_anomaly_mask=True,
                label_smoothing=True, alpha=0.1, beta=0.01,
            )
        live_model = ConvAEC(m_params)
        live_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        live_model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(y_live.T[None, :, :], dtype=torch.float32)  # (1, W, D)
            x_norm_tensor = torch.tensor(z_live.T[None, :, :], dtype=torch.float32)
            # Ghép 2 mẫu để BatchNorm hoạt động ổn định ở mọi cấu hình
            batch_in = torch.cat([x_norm_tensor, x_tensor], dim=0)
            rec_out, cls_out, _ = live_model(batch_in)

            rec_norm = rec_out[0, :, 0].numpy()
            rec_inj = rec_out[1, :, 0].numpy()
            prob_norm = cls_out[0].numpy()
            prob_inj = cls_out[1].numpy()

        mse_norm_val = float(np.mean((z_live[0] - rec_norm) ** 2))
        mse_inj_val = float(np.mean((y_live[0] - rec_inj) ** 2))

        k1, k2, k3 = st.columns(3)
        k1.metric("Sai số tái tạo MSE (Cửa sổ Gốc)", f"{mse_norm_val:.5f}")
        k2.metric(
            f"Sai số tái tạo MSE (Sau khi bơm {live_aug_type.upper()})",
            f"{mse_inj_val:.5f}",
            delta=f"{mse_inj_val - mse_norm_val:+.5f}",
            delta_color="inverse",
        )
        pred_top_cls = ANOMALY_TYPES[int(np.argmax(prob_inj[1:])) + 1] if live_aug_type != 'normal' else ANOMALY_TYPES[int(np.argmax(prob_inj))]
        k3.metric("Lớp Bất thường Dự đoán Cao nhất", pred_top_cls.upper())

        col_l1, col_l2 = st.columns([1.3, 1.0])
        with col_l1:
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(y=z_live[0], name="Tín hiệu gốc Z", line=dict(color="#7f8c8d", width=1.8, dash="dot")))
            fig_live.add_trace(go.Scatter(y=y_live[0], name=f"Tín hiệu sau khi bơm [{live_aug_type.upper()}]", line=dict(color="#c0392b", width=2.6)))
            fig_live.add_trace(go.Scatter(y=rec_inj, name="Tái tạo của Decoder X̂", line=dict(color="#2980b9", width=2.0)))
            apts_live = np.where(mask_live[0] == 0)[0]
            if len(apts_live) > 0:
                fig_live.add_vrect(
                    x0=int(apts_live.min()), x1=int(apts_live.max()) + 1,
                    fillcolor="rgba(231, 76, 60, 0.20)", line_width=0,
                    annotation_text="Vùng chèn đột biến"
                )
            fig_live.update_layout(
                title="Kết quả Tái tạo Trực tiếp từ Checkpoint bestmodel.pkl",
                xaxis_title="Timestep trong cửa sổ (L=100)",
                yaxis_title="Giá trị",
                height=360,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_live, use_container_width=True)

        with col_l2:
            fig_live_cls = go.Figure()
            fig_live_cls.add_trace(go.Bar(
                x=ANOMALY_TYPES, y=prob_norm, name="Trước khi bơm (Gốc)",
                marker_color="rgba(149, 165, 166, 0.55)"
            ))
            fig_live_cls.add_trace(go.Bar(
                x=ANOMALY_TYPES, y=prob_inj, name=f"Sau khi bơm {live_aug_type.upper()}",
                marker_color=[CLASS_COLORS[c] for c in ANOMALY_TYPES]
            ))
            fig_live_cls.update_layout(
                title="Phản ứng của Nhánh Multiclass Classifier (12 lớp)",
                yaxis_title="Xác suất Softmax",
                barmode="group",
                height=360,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_live_cls, use_container_width=True)
    else:
        st.warning("Thư mục thực nghiệm này chưa có file `bestmodel.pkl` để suy luận trực tiếp.")
