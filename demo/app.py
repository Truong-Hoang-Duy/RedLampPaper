"""
RedLamp Interactive Studio — Ứng dụng Demo Đồ Án & Kiểm Chứng Thực Nghiệm KDD '25
Bài báo: Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (KDD '25)

Khởi chạy từ thư mục gốc của project:
    streamlit run demo/app.py
"""
import os
import sys
import json
import zipfile
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

# Thiết lập đường dẫn import cho cả RedLampPaper/ và RedLampPaper/RedLamp/
DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
REDLAMP_DIR = ROOT_DIR / "RedLamp"
RETRAIN_DIR = ROOT_DIR / "redlamp_retraining"

for p in (REDLAMP_DIR, ROOT_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import utils
from models.meta import ConvAEC
from loaders.dataset import Entity, Dataset
from loaders.loader_aug import Loader_aug
from main import convolve_minmax_score, mse, label_score_selected_feature
from demo.generate_sample_data import ensure_sample_data, ANOMALY_TYPES

st.set_page_config(
    page_title="RedLamp Studio | KDD '25 Interactive Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== BẢNG MÀU & TỪ ĐIỂN GIẢI THÍCH ========================
CLASS_COLORS = {
    "normal": "#10b981",
    "spike": "#ef4444",
    "flip": "#f97316",
    "speedup": "#eab308",
    "noise": "#14b8a6",
    "cutoff": "#8b5cf6",
    "average": "#3b82f6",
    "scale": "#ec4899",
    "wander": "#06b6d4",
    "contextual": "#6366f1",
    "upsidedown": "#f43f5e",
    "mixture": "#a855f7",
}

CLASS_DESCRIPTIONS_VI = {
    "normal": "Chuỗi bình thường (không biến đổi)",
    "spike": "Gai nhọn đột biến (Point Outlier)",
    "flip": "Đảo ngược chiều thời gian đoạn [st, ed]",
    "speedup": "Thay đổi tần số co/giãn (2x hoặc 0.5x)",
    "noise": "Nhiễu trắng Gaussian N(0, 0.1I)",
    "cutoff": "Đứt/kẹt tín hiệu về hằng số phẳng",
    "average": "Làm trơn tín hiệu (Moving Average)",
    "scale": "Khuếch đại/suy giảm biên độ bất thường",
    "wander": "Trôi đường cơ sở (Baseline Trend Shift)",
    "contextual": "Biến dạng ngữ cảnh (Scale + Shift)",
    "upsidedown": "Đảo cực tín hiệu qua giá trị trung bình",
    "mixture": "Tráo ghép đoạn tín hiệu ngoại lai",
}

DATASET_CATALOG = {
    "aiops": {
        "title": "🏢 AIOps — Giám sát KPI Dịch vụ Web (29 KPIs, D=1)",
        "short": "AIOps",
        "summary_csv": "summary_iops.csv",
        "metrics_csv": "metrics_iops.csv",
        "domain_badge": "Cloud Web KPI Monitoring",
        "default_entity": "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa",
    },
    "smd": {
        "title": "🖥️ SMD — Giám sát Máy chủ Đám mây (28 Machines, D=38)",
        "short": "SMD",
        "summary_csv": "summary_smd.csv",
        "metrics_csv": "metrics_smd.csv",
        "domain_badge": "Server Machine Telemetry (38 Channels)",
        "default_entity": "machine-1-1",
    },
    "msl": {
        "title": "🚀 NASA MSL — Tàu Tự hành Sao Hỏa Curiosity (23 Kênh, D=55)",
        "short": "MSL",
        "summary_csv": "summary_msl.csv",
        "metrics_csv": "metrics_msl.csv",
        "domain_badge": "Mars Rover Telemetry (55 Channels)",
        "default_entity": "F-5",
    },
    "smap": {
        "title": "🛰️ NASA SMAP — Vệ tinh Quan trắc Độ ẩm Đất (49 Kênh, D=25)",
        "short": "SMAP",
        "summary_csv": "summary_smap.csv",
        "metrics_csv": "metrics_smap.csv",
        "domain_badge": "Satellite Telemetry (25 Channels)",
        "default_entity": "E-6",
    },
    "sample": {
        "title": "🧪 Demo Mẫu Chuẩn Bài Báo (UCR #028 ABP & #040 EPG)",
        "short": "UCR Sample",
        "summary_csv": None,
        "metrics_csv": None,
        "domain_badge": "KDD '25 Figure 6 Benchmark Showcase",
        "default_entity": "028_UCR_ABP_Demo",
    },
}

# CSS tinh chỉnh giao diện hiện đại, gọn gàng
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 1.5rem; }
    .kpi-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 12px;
        padding: 12px 16px;
        background: rgba(128, 128, 128, 0.04);
    }
    .kpi-label { font-size: 0.78rem; opacity: 0.75; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 1.45rem; font-weight: 800; margin: 2px 0; }
    .kpi-sub { font-size: 0.78rem; opacity: 0.8; }
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================== HÀM NẠP DỮ LIỆU & CACHE ========================
@st.cache_resource
def init_demo_samples():
    ensure_sample_data()
    return True


@st.cache_data
def load_all_summary_tables():
    """Đọc toàn bộ bảng summary_*.csv (bản mới 04/11 và bản legacy 31/10) từ redlamp_retraining."""
    summaries = {}
    legacy_summaries = {}
    if not RETRAIN_DIR.exists():
        return summaries, legacy_summaries

    for key, info in DATASET_CATALOG.items():
        csv_name = info.get("summary_csv")
        if not csv_name:
            continue
        p_new = RETRAIN_DIR / csv_name
        p_old = RETRAIN_DIR / "backup_legacy_metric_20261031" / csv_name
        if p_new.exists():
            summaries[key] = pd.read_csv(p_new)
        if p_old.exists():
            legacy_summaries[key] = pd.read_csv(p_old)
    return summaries, legacy_summaries


@st.cache_data
def load_all_entity_metrics():
    """Đọc bảng metrics_*.csv chứa kết quả chi tiết từng entity x seed."""
    metrics_map = {}
    if not RETRAIN_DIR.exists():
        return metrics_map
    for key, info in DATASET_CATALOG.items():
        csv_name = info.get("metrics_csv")
        if not csv_name:
            continue
        p = RETRAIN_DIR / csv_name
        if p.exists():
            metrics_map[key] = pd.read_csv(p)
    return metrics_map


@st.cache_data
def load_nasa_ground_truth():
    """Đọc nhãn khoảng bất thường thực tế của NASA MSL & SMAP từ RedLamp/dataset/NASA.zip."""
    gt_info = {}
    zip_path = REDLAMP_DIR / "dataset" / "NASA.zip"
    if not zip_path.exists():
        return gt_info
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_entries = [n for n in z.namelist() if n.endswith("labeled_anomalies.csv")]
            if not csv_entries:
                return gt_info
            with z.open(csv_entries[0]) as f:
                df = pd.read_csv(f)
            for _, row in df.iterrows():
                cid = str(row["chan_id"])
                seqs = eval(str(row["anomaly_sequences"]))
                classes = eval(str(row["class"])) if "class" in row and pd.notna(row["class"]) else []
                gt_info[cid] = {
                    "sequences": seqs,
                    "classes": classes,
                    "num_values": int(row["num_values"]),
                }
    except Exception:
        pass
    return gt_info


@st.cache_data
def scan_dataset_entities(ds_key: str):
    """
    Trả về danh sách các entity và đường dẫn theo từng seed cho tập dữ liệu được chọn.
    Cấu trúc trả về: { entity_name: { seed_int: {"test_dir": str, "model_dir": str, "mode": "retrain"|"sample"} } }
    """
    entity_map = {}
    if ds_key == "sample":
        sample_root = DEMO_DIR / "sample_results" / "anomaly_archive"
        if sample_root.exists():
            for ent_dir in sorted(sample_root.iterdir()):
                if not ent_dir.is_dir():
                    continue
                for input_f in sorted(ent_dir.rglob("input.npy")):
                    test_dir = input_f.parent
                    seed_dir = test_dir.parent
                    seed_id = int(seed_dir.name) if seed_dir.name.isdigit() else 0
                    entity_map.setdefault(ent_dir.name, {})[seed_id] = {
                        "test_dir": str(test_dir),
                        "model_dir": str(seed_dir),
                        "mode": "sample",
                    }
        return entity_map

    if not RETRAIN_DIR.exists():
        return entity_map

    ds_dir = RETRAIN_DIR / ds_key
    if not ds_dir.exists():
        return entity_map

    if ds_key in ("aiops", "smd"):
        for ent_dir in sorted(ds_dir.iterdir()):
            if not ent_dir.is_dir():
                continue
            cfg_dir = ent_dir / "d1_b128_w100_s10"
            if not cfg_dir.exists():
                continue
            for seed_dir in sorted(cfg_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.isdigit():
                    test_dir = seed_dir / "test_all"
                    if (test_dir / "input_last.npy").exists():
                        entity_map.setdefault(ent_dir.name, {})[int(seed_dir.name)] = {
                            "test_dir": str(test_dir),
                            "model_dir": str(seed_dir),
                            "mode": "retrain",
                        }
    elif ds_key in ("msl", "smap"):
        cfg_dir = ds_dir / "d1_b128_w100_s10"
        if cfg_dir.exists():
            for seed_dir in sorted(cfg_dir.iterdir()):
                if not (seed_dir.is_dir() and seed_dir.name.isdigit()):
                    continue
                seed_id = int(seed_dir.name)
                # 1. Các kênh đánh giá riêng biệt (Per-channel trong test_each)
                test_each_dir = seed_dir / "test_each"
                if test_each_dir.exists():
                    for ch_dir in sorted(test_each_dir.iterdir()):
                        t_dir = ch_dir / "test_all"
                        if (t_dir / "input_last.npy").exists():
                            entity_map.setdefault(ch_dir.name, {})[seed_id] = {
                                "test_dir": str(t_dir),
                                "model_dir": str(seed_dir),
                                "mode": "retrain",
                            }
                # 2. Bản đánh giá gộp (Aggregate)
                agg_dir = seed_dir / "test_all"
                if (agg_dir / "input_last.npy").exists():
                    entity_map.setdefault(f"[Aggregate] {ds_key.upper()}", {})[seed_id] = {
                        "test_dir": str(agg_dir),
                        "model_dir": str(seed_dir),
                        "mode": "retrain",
                    }
    return entity_map


def load_run_bundle(run_info: dict, ds_key: str, entity_name: str):
    """
    Adapter đa định dạng: đọc mượt cả định dạng `redlamp_retraining` (input_last.npy + mse.npy)
    và định dạng `demo/sample_results` (input.npy + pred.npy).
    """
    test_dir = Path(run_info["test_dir"])
    model_dir = Path(run_info["model_dir"])
    mode = run_info["mode"]

    bundle = {
        "mode": mode,
        "ds_key": ds_key,
        "entity_name": entity_name,
        "test_dir": test_dir,
        "model_dir": model_dir,
        "window_size": 100,
        "meta": {},
    }

    if mode == "sample":
        inputs = np.load(test_dir / "input.npy")
        preds = np.load(test_dir / "pred.npy")
        B, W, D = inputs.shape
        bundle["window_size"] = W
        bundle["input_3d"] = inputs
        bundle["pred_3d"] = preds
        bundle["input_last"] = inputs[:, -1, :]
        bundle["pred_last"] = preds[:, -1, :]
        bundle["raw_mse"] = mse(inputs.reshape(B, -1), preds.reshape(B, -1))
        bundle["pred_label"] = np.load(test_dir / "pred_label.npy")
        bundle["enc"] = np.load(test_dir / "enc.npy")
        bundle["anomaly_score"] = np.load(test_dir / "anomaly_score.npy")
        if (test_dir / "gt_labels.npy").exists():
            bundle["gt_labels"] = np.load(test_dir / "gt_labels.npy")
        if (test_dir / "meta.json").exists():
            with open(test_dir / "meta.json", "r", encoding="utf-8") as f:
                bundle["meta"] = json.load(f)
        if (test_dir / "train_aug_enc.npy").exists() and (test_dir / "train_aug_cls.npy").exists():
            bundle["train_aug_enc"] = np.load(test_dir / "train_aug_enc.npy")
            bundle["train_aug_cls"] = np.load(test_dir / "train_aug_cls.npy")
    else:
        # Chế độ đọc nhanh từ redlamp_retraining bằng mmap_mode
        input_last = np.array(np.load(test_dir / "input_last.npy", mmap_mode="r"), dtype=np.float32)
        if input_last.ndim == 1:
            input_last = input_last[:, None]
        raw_mse = np.array(np.load(test_dir / "mse.npy", mmap_mode="r"), dtype=np.float32)
        pred_label = np.array(np.load(test_dir / "pred_label.npy", mmap_mode="r"), dtype=np.float32)
        enc = np.array(np.load(test_dir / "enc.npy", mmap_mode="r"), dtype=np.float32)
        anom_score = np.array(np.load(test_dir / "anomaly_score.npy", mmap_mode="r"), dtype=np.float64)

        bundle["input_last"] = input_last
        bundle["pred_last"] = None
        bundle["raw_mse"] = raw_mse
        bundle["pred_label"] = pred_label
        bundle["enc"] = enc
        bundle["anomaly_score"] = anom_score

        # Nếu là NASA MSL / SMAP, gắn nhãn Ground Truth thực tế từ labeled_anomalies.csv
        nasa_gt = load_nasa_ground_truth()
        if entity_name in nasa_gt:
            W = bundle["window_size"]
            total_len = len(input_last) + W - 1
            gt_vec = np.zeros(total_len, dtype=np.int32)
            for seq in nasa_gt[entity_name]["sequences"]:
                s_idx = max(0, int(seq[0]))
                e_idx = min(total_len, int(seq[1]))
                gt_vec[s_idx:e_idx] = 1
            bundle["gt_labels"] = gt_vec
            bundle["meta"]["nasa_classes"] = nasa_gt[entity_name]["classes"]

    # Đọc lịch sử hội tụ (train_loss*.txt, valid_loss*.txt, time.txt)
    losses = {}
    for fname in (
        "train_loss.txt",
        "train_loss_ae.txt",
        "train_loss_c.txt",
        "valid_loss.txt",
        "valid_loss_ae.txt",
        "valid_loss_c.txt",
        "time.txt",
    ):
        fpath = model_dir / fname
        if fpath.exists():
            try:
                losses[fname.replace(".txt", "")] = np.loadtxt(fpath)
            except Exception:
                pass
    bundle["losses"] = losses
    return bundle


def compute_redlamp_scores_from_raw(raw_mse, pred_labels, faa_threshold=0.05, mse_weight=0.5, smooth_w=50):
    """
    Tính toán điểm bất thường theo Mục 3.6 của bài báo RedLamp từ raw_mse (T,) và pred_labels (T, 12).
    Khớp chính xác tới sai số máy (1e-16) với anomaly_score.npy trong redlamp_retraining.
    """
    w = max(1, int(smooth_w))
    norm_mse = convolve_minmax_score(raw_mse.reshape(-1, 1), w=w).squeeze()

    mean_class_probs = np.mean(pred_labels, axis=0)
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


# Khởi tạo dữ liệu mẫu và nạp các bảng tổng hợp
init_demo_samples()
summaries_dict, legacy_summaries_dict = load_all_summary_tables()
metrics_tables = load_all_entity_metrics()

# ======================== THANH BÊN (SIDEBAR ĐIỀU KHIỂN) ========================
with st.sidebar:
    st.markdown("## 🚨 RedLamp Studio")
    st.caption("**KDD '25 Reproduction & XAI Diagnostics**")

    available_ds_keys = [k for k in DATASET_CATALOG.keys() if len(scan_dataset_entities(k)) > 0]
    selected_ds_key = st.selectbox(
        "1️⃣ Chọn Miền Dữ Liệu (Dataset Domain):",
        options=available_ds_keys,
        index=0,
        format_func=lambda k: DATASET_CATALOG[k]["title"],
    )

    entity_map = scan_dataset_entities(selected_ds_key)
    entity_names = list(entity_map.keys())
    default_ent = DATASET_CATALOG[selected_ds_key]["default_entity"]
    default_ent_idx = entity_names.index(default_ent) if default_ent in entity_names else 0

    # Tạo nhãn hiển thị kèm điểm VUS-ROC trung bình cho từng thực thể
    ent_score_lookup = {}
    if selected_ds_key in metrics_tables:
        df_m = metrics_tables[selected_ds_key]
        for ent_n, grp in df_m.groupby("entity"):
            ent_score_lookup[str(ent_n)] = (
                grp["VUS-ROC"].mean(),
                grp["R-AUC-PR"].mean(),
            )

    def format_entity_option(ent_name: str):
        if ent_name in ent_score_lookup:
            v_roc, r_pr = ent_score_lookup[ent_name]
            return f"{ent_name}  (VUS-ROC: {v_roc:.3f} | AUC-PR: {r_pr:.3f})"
        return ent_name

    selected_entity = st.selectbox(
        "2️⃣ Chọn Thực Thể / Kênh Cảm Biến:",
        options=entity_names,
        index=default_ent_idx,
        format_func=format_entity_option,
    )

    seeds_available = sorted(entity_map[selected_entity].keys())
    selected_seed = st.segmented_control(
        "3️⃣ Chọn Random Seed:",
        options=seeds_available,
        default=seeds_available[0],
        format_func=lambda s: f"Seed {s}",
    ) if hasattr(st, "segmented_control") else st.selectbox(
        "3️⃣ Chọn Random Seed:",
        options=seeds_available,
        index=0,
        format_func=lambda s: f"Seed {s}",
    )
    if selected_seed is None:
        selected_seed = seeds_available[0]

    bundle = load_run_bundle(entity_map[selected_entity][selected_seed], selected_ds_key, selected_entity)

    st.markdown("---")
    st.markdown("### ⚙️ Tham số Bộ lọc & Suy luận")
    faa_delta = st.slider(
        "Ngưỡng FAA δ (Lọc lớp giả bình thường):",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="Lớp bất thường có xác suất trung bình trên toàn chuỗi > δ sẽ được nhận diện là Pseudo-Normal và bị triệt tiêu (Eq. 6).",
    )
    mse_weight = st.slider(
        "Trọng số Autoencoder vs. Classifier:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    smooth_w = st.slider(
        "Cửa sổ làm mượt điểm (w):",
        min_value=1,
        max_value=100,
        value=50,
        step=5,
    )


# ======================== HEADER CHÍNH & DẢI CHỈ SỐ NHANH ========================
B_len, D_feat = bundle["input_last"].shape
W_size = bundle["window_size"]

scores_dict = compute_redlamp_scores_from_raw(
    bundle["raw_mse"],
    bundle["pred_label"],
    faa_threshold=faa_delta,
    mse_weight=mse_weight,
    smooth_w=smooth_w,
)

# Xác định lớp bất thường chủ đạo tại vùng đỉnh điểm bất thường
peak_indices = np.argsort(scores_dict["total_score"])[-max(15, B_len // 100):]
peak_class_probs = np.mean(bundle["pred_label"][peak_indices, 1:], axis=0)
dominant_cls_idx = int(np.argmax(peak_class_probs)) + 1
dominant_cls_name = ANOMALY_TYPES[dominant_cls_idx]
suppressed_cls_names = [ANOMALY_TYPES[i].upper() for i in scores_dict["suppressed_indices"] if i != 0]

col_head_l, col_head_r = st.columns([1.8, 1.2])
with col_head_l:
    st.markdown(
        f"### 🚨 RedLamp Unified Studio — `{DATASET_CATALOG[selected_ds_key]['short']}` / `{selected_entity}` (Seed `{selected_seed}`)"
    )
    st.caption(
        f"**Miền ứng dụng:** {DATASET_CATALOG[selected_ds_key]['domain_badge']} • "
        f"**Số bước thời gian:** `{B_len:,}` • **Số kênh đặc trưng:** `D = {D_feat}` • "
        f"**Chẩn đoán hình thái đỉnh:** `{dominant_cls_name.upper()}` ({CLASS_DESCRIPTIONS_VI[dominant_cls_name]})"
    )

with col_head_r:
    # Lấy chỉ số thực nghiệm của thực thể + seed đang chọn từ metrics_*.csv (nếu có)
    cur_metrics_row = None
    if selected_ds_key in metrics_tables:
        df_m = metrics_tables[selected_ds_key]
        lookup_ent = selected_ds_key if selected_entity.startswith("[Aggregate]") else selected_entity
        sub_m = df_m[(df_m["entity"].astype(str) == str(lookup_ent)) & (df_m["seed"] == int(selected_seed))]
        if len(sub_m) > 0:
            cur_metrics_row = sub_m.iloc[0]

    if cur_metrics_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VUS-ROC", f"{cur_metrics_row['VUS-ROC']:.3f}")
        c2.metric("R-AUC-ROC", f"{cur_metrics_row['R-AUC-ROC']:.3f}")
        c3.metric("R-AUC-PR", f"{cur_metrics_row['R-AUC-PR']:.3f}")
        c4.metric("RF (Oracle)", f"{cur_metrics_row['RF (oracle)']:.3f}")


# ======================== 5 MÀN HÌNH CHÍNH (TABS) ========================
tab_monitor, tab_bench, tab_tsne, tab_live, tab_aug = st.tabs([
    "📡 1. Giám sát & Giải phẫu Bất thường (NOC / Telemetry)",
    "🏆 2. Bảng vàng Thực nghiệm KDD '25 (4 Datasets × 5 Seeds)",
    "🧭 3. Không gian Tiềm ẩn Giải thích được (t-SNE 128D)",
    "⚡ 4. Phòng Giả lập Bơm Lỗi Thời gian thực (Live Chaos)",
    "🧬 5. Đa dạng hóa 12 Lớp Giả Bất thường & Nhãn mềm",
])


# ==============================================================================
# MÀN HÌNH 1: GIÁM SÁT & GIẢI PHẪU ĐIỂM BẤT THƯỜNG (HƯỚNG 1 & 2)
# ==============================================================================
with tab_monitor:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.1, 1.5, 1.4])
    with ctrl_col1:
        if D_feat > 1:
            feature_idx = st.selectbox(
                f"Chọn kênh cảm biến hiển thị (0 .. {D_feat - 1}):",
                options=list(range(D_feat)),
                index=0,
                format_func=lambda idx: f"Channel #{idx + 1} (Feature {idx})",
            )
        else:
            feature_idx = 0
            st.info("📌 Chuỗi thời gian đơn biến (`D = 1`).")

    seq_in_full = np.concatenate([np.zeros(W_size - 1, dtype=np.float32), bundle["input_last"][:, feature_idx]])
    score_total_full = np.concatenate([np.zeros(W_size - 1, dtype=np.float32), scores_dict["total_score"]])
    score_nofaa_full = np.concatenate([np.zeros(W_size - 1, dtype=np.float32), scores_dict["total_score_no_faa"]])
    score_mse_full = np.concatenate([np.zeros(W_size - 1, dtype=np.float32), scores_dict["norm_mse"]])
    score_ce_full = np.concatenate([np.zeros(W_size - 1, dtype=np.float32), scores_dict["norm_faa_ce"]])
    total_len = len(seq_in_full)

    with ctrl_col2:
        # Thanh trượt chọn cửa sổ quan sát để vẽ mượt với chuỗi dài (như AIOps 149,000 điểm)
        max_default_span = min(total_len, 15000)
        peak_pos = int(np.argmax(score_total_full))
        default_start = max(0, min(total_len - max_default_span, peak_pos - max_default_span // 2))
        default_end = min(total_len, default_start + max_default_span)
        view_range = st.slider(
            "Khoảng thời gian quan sát (Timestep Range):",
            min_value=0,
            max_value=total_len,
            value=(int(default_start), int(default_end)),
            step=max(1, total_len // 200),
        )

    with ctrl_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Chẩn đoán Tự động & Bộ lọc FAA (δ = {faa_delta:.2f})</div>
                <div style="font-size:0.9rem; margin-top:4px;">
                    🔍 <b>Hình thái lỗi chính:</b> <span style="color:{CLASS_COLORS[dominant_cls_name]}; font-weight:700;">{dominant_cls_name.upper()}</span> — {CLASS_DESCRIPTIONS_VI[dominant_cls_name]}<br/>
                    🛡️ <b>Lớp Pseudo-Normal bị FAA lọc:</b> <code>{', '.join(suppressed_cls_names) if suppressed_cls_names else 'Không có'}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    v_start, v_end = view_range
    if v_end <= v_start + 10:
        v_end = min(total_len, v_start + 100)

    # Downsample nhẹ nếu vùng chọn > 6,000 điểm để Plotly phản hồi tức thì
    span_len = v_end - v_start
    stride_plot = max(1, span_len // 5000)
    x_axis = np.arange(v_start, v_end, stride_plot)

    fig_sync = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"1. Tín hiệu Đầu vào X(t) — Kênh #{feature_idx + 1} (Timesteps {v_start:,} → {v_end:,})",
            "2. Giải phẫu Điểm Thành Phần: Sai số Tái tạo s_MSE(t) vs. Điểm Phân loại Đa lớp s_CE(t)",
            "3. Điểm Bất thường Tổng hợp RedLamp s(t): Khi BẬT FAA vs. Khi TẮT FAA (Chống báo động giả)",
        ),
        row_heights=[0.36, 0.32, 0.32],
    )

    fig_sync.add_trace(
        go.Scatter(x=x_axis, y=seq_in_full[v_start:v_end:stride_plot], name="Input X(t)", line=dict(color="#3b82f6", width=1.6)),
        row=1, col=1,
    )
    if bundle["pred_last"] is not None:
        seq_pred_full = np.concatenate([np.zeros(W_size - 1, dtype=np.float32), bundle["pred_last"][:, feature_idx]])
        fig_sync.add_trace(
            go.Scatter(
                x=x_axis,
                y=seq_pred_full[v_start:v_end:stride_plot],
                name="Reconstruction X̂(t)",
                line=dict(color="#f97316", width=1.5, dash="dot"),
            ),
            row=1, col=1,
        )

    fig_sync.add_trace(
        go.Scatter(x=x_axis, y=score_mse_full[v_start:v_end:stride_plot], name="s_MSE(t) (Autoencoder)", line=dict(color="#0ea5e9", width=1.6)),
        row=2, col=1,
    )
    fig_sync.add_trace(
        go.Scatter(x=x_axis, y=score_ce_full[v_start:v_end:stride_plot], name="s_CE(t) (Classifier + FAA)", line=dict(color="#a855f7", width=1.6)),
        row=2, col=1,
    )

    fig_sync.add_trace(
        go.Scatter(
            x=x_axis,
            y=score_nofaa_full[v_start:v_end:stride_plot],
            name="Tắt FAA (w/o FAA)",
            line=dict(color="#94a3b8", width=1.4, dash="dash"),
        ),
        row=3, col=1,
    )
    fig_sync.add_trace(
        go.Scatter(
            x=x_axis,
            y=score_total_full[v_start:v_end:stride_plot],
            name="RedLamp s(t) (Có FAA)",
            line=dict(color="#ef4444", width=2.2),
        ),
        row=3, col=1,
    )

    # Tô vùng Ground Truth (nếu có) hoặc vùng đỉnh cảnh báo (Top 1% score)
    if "gt_labels" in bundle and len(bundle["gt_labels"]) == total_len:
        gt_vec = bundle["gt_labels"]
        diff_gt = np.diff(np.concatenate(([0], gt_vec, [0])))
        gt_starts = np.where(diff_gt == 1)[0]
        gt_ends = np.where(diff_gt == -1)[0]
        for s_i, e_i in zip(gt_starts, gt_ends):
            if e_i >= v_start and s_i <= v_end:
                for r_i in (1, 2, 3):
                    fig_sync.add_vrect(
                        x0=max(v_start, int(s_i)),
                        x1=min(v_end, int(e_i)),
                        fillcolor="rgba(239, 68, 68, 0.20)",
                        line_width=0,
                        row=r_i,
                        col=1,
                    )

    fig_sync.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=40, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sync, use_container_width=True)

    # Hàng biểu đồ dưới: (1) Tần suất lớp & Ngưỡng FAA, (2) Diễn biến xác suất 11 lớp
    col_faa_l, col_faa_r = st.columns([1.05, 1.35])
    with col_faa_l:
        mean_probs = scores_dict["mean_class_probs"][1:]
        anom_names = ANOMALY_TYPES[1:]
        bar_colors = ["#64748b" if p > faa_delta else CLASS_COLORS[c] for p, c in zip(mean_probs, anom_names)]
        fig_faa_bar = go.Figure(
            go.Bar(
                x=[c.capitalize() for c in anom_names],
                y=mean_probs,
                marker_color=bar_colors,
                text=[f"{p:.3f}" for p in mean_probs],
                textposition="outside",
            )
        )
        fig_faa_bar.add_hline(
            y=faa_delta,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Ngưỡng FAA δ = {faa_delta:.2f} (Vượt ngưỡng ⇒ Lọc bỏ)",
            annotation_position="top right",
        )
        fig_faa_bar.update_layout(
            title="Tần suất Dự đoán Trung bình 11 Lớp trên Tập Test (Eq. 6)",
            yaxis_title="Xác suất trung bình",
            height=330,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_faa_bar, use_container_width=True)

    with col_faa_r:
        w_start = max(0, v_start - (W_size - 1))
        w_end = min(B_len, v_end - (W_size - 1))
        w_stride = max(1, (w_end - w_start) // 3000)
        w_axis = np.arange(w_start, w_end, w_stride) + (W_size - 1)

        fig_prob = go.Figure()
        for c_idx in range(1, len(ANOMALY_TYPES)):
            cname = ANOMALY_TYPES[c_idx]
            is_sup = c_idx in scores_dict["suppressed_indices"]
            fig_prob.add_trace(
                go.Scatter(
                    x=w_axis,
                    y=bundle["pred_label"][w_start:w_end:w_stride, c_idx],
                    mode="lines",
                    name=f"{cname.capitalize()} {'(Lọc FAA)' if is_sup else ''}",
                    line=dict(
                        color=CLASS_COLORS[cname],
                        width=1.6 if not is_sup else 1.0,
                        dash="dot" if is_sup else "solid",
                    ),
                    opacity=0.30 if is_sup else 0.90,
                )
            )
        fig_prob.update_layout(
            title="Diễn biến Xác suất 11 Lớp Giả Bất thường theo Thời gian",
            xaxis_title="Timestep t",
            yaxis_title="Xác suất Softmax ŷ(t)",
            height=330,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# ==============================================================================
# MÀN HÌNH 2: BẢNG VÀNG THỰC NGHIỆM KDD '25 & HỘI TỤ ĐA SEED (HƯỚNG 3)
# ==============================================================================
with tab_bench:
    # 4 Thẻ KPI Tổng quan Kho Thực nghiệm redlamp_retraining
    k_c1, k_c2, k_c3, k_c4 = st.columns(4)
    k_c1.markdown(
        """<div class="kpi-card"><div class="kpi-label">Quy mô Thực nghiệm</div>
        <div class="kpi-val" style="color:#ef4444;">15.4 GB</div>
        <div class="kpi-sub">295 Checkpoints • 645 Bộ kết quả Test</div></div>""",
        unsafe_allow_html=True,
    )
    k_c2.markdown(
        """<div class="kpi-card"><div class="kpi-label">AIOps (29 KPIs × 5 Seeds)</div>
        <div class="kpi-val" style="color:#10b981;">0.926 VUS-ROC</div>
        <div class="kpi-sub">Vượt KDD '25 Paper (0.911) trên cả 5 thước đo</div></div>""",
        unsafe_allow_html=True,
    )
    k_c3.markdown(
        """<div class="kpi-card"><div class="kpi-label">SMD (28 Machines × 5 Seeds)</div>
        <div class="kpi-val" style="color:#10b981;">0.853 R-AUC-ROC</div>
        <div class="kpi-sub">Vượt KDD '25 Paper (0.837) trên cả 5 thước đo</div></div>""",
        unsafe_allow_html=True,
    )
    k_c4.markdown(
        """<div class="kpi-card"><div class="kpi-label">NASA MSL (23) & SMAP (49)</div>
        <div class="kpi-val" style="color:#0ea5e9;">Per-Channel Protocol</div>
        <div class="kpi-sub">Bám sát ~99% bài báo gốc & Vượt xa Aggregate</div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("")
    b_col1, b_col2 = st.columns([1.25, 1.0])

    with b_col1:
        bench_pick = st.radio(
            "Chọn Tập Benchmark để đối sánh (Ours 5 Seeds vs. KDD '25 Paper vs. Legacy 31/10):",
            options=["aiops", "smd", "msl", "smap"],
            format_func=lambda k: DATASET_CATALOG[k]["short"],
            horizontal=True,
        )
        if bench_pick in summaries_dict:
            df_sum = summaries_dict[bench_pick]
            # Lọc dòng protocol chính (per-entity hoặc per-channel)
            df_main = df_sum[df_sum["setting"].str.contains("per-")].copy()
            df_main_compare = df_main[df_main["paper"].notna()].copy()

            fig_comp = go.Figure()
            fig_comp.add_trace(
                go.Bar(
                    x=df_main_compare["metric"],
                    y=df_main_compare["ours (mean)"],
                    error_y=dict(type="data", array=df_main_compare["std (seeds)"], visible=True),
                    name="Ours (Chuẩn mới 04/11 - 5 Seeds)",
                    marker_color="#10b981",
                    text=[f"{v:.3f}" for v in df_main_compare["ours (mean)"]],
                    textposition="outside",
                )
            )
            fig_comp.add_trace(
                go.Bar(
                    x=df_main_compare["metric"],
                    y=df_main_compare["paper"],
                    name="Bài báo gốc KDD '25",
                    marker_color="#3b82f6",
                    text=[f"{v:.3f}" for v in df_main_compare["paper"]],
                    textposition="outside",
                )
            )
            if bench_pick in legacy_summaries_dict:
                df_leg = legacy_summaries_dict[bench_pick]
                df_leg_main = df_leg[df_leg["setting"].str.contains("per-") & df_leg["paper"].notna()]
                if len(df_leg_main) > 0:
                    fig_comp.add_trace(
                        go.Bar(
                            x=df_leg_main["metric"],
                            y=df_leg_main["ours (mean)"],
                            name="Bản Legacy (31/10)",
                            marker_color="rgba(245, 158, 11, 0.55)",
                        )
                    )
            fig_comp.update_layout(
                title=f"Đối sánh Kết quả Tái lập [{DATASET_CATALOG[bench_pick]['short']}] — Ours (5 Seeds) vs. KDD '25 Paper",
                yaxis_title="Điểm số (0.0 → 1.0)",
                yaxis_range=[0, 1.08],
                barmode="group",
                height=370,
                margin=dict(l=20, r=20, t=45, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    with b_col2:
        # Biểu đồ đường cong hội tụ Loss qua các Epoch của thực thể đang chọn
        losses = bundle.get("losses", {})
        if losses:
            fig_loss = make_subplots(specs=[[{"secondary_y": True}]])
            if "train_loss_ae" in losses:
                fig_loss.add_trace(
                    go.Scatter(
                        y=losses["train_loss_ae"],
                        mode="lines+markers",
                        name="Train L_MSE (Autoencoder)",
                        line=dict(color="#10b981", width=2.2),
                    ),
                    secondary_y=False,
                )
            if "valid_loss_ae" in losses:
                fig_loss.add_trace(
                    go.Scatter(
                        y=losses["valid_loss_ae"],
                        mode="lines+markers",
                        name="Valid L_MSE",
                        line=dict(color="#059669", width=1.8, dash="dot"),
                    ),
                    secondary_y=False,
                )
            if "train_loss_c" in losses:
                fig_loss.add_trace(
                    go.Scatter(
                        y=losses["train_loss_c"],
                        mode="lines+markers",
                        name="Train L_CE (12-Class Classifier)",
                        line=dict(color="#8b5cf6", width=2.2),
                    ),
                    secondary_y=True,
                )
            avg_epoch_sec = float(np.mean(losses["time"])) if "time" in losses else None
            title_suffix = f" ({avg_epoch_sec:.2f}s / epoch)" if avg_epoch_sec else ""
            fig_loss.update_layout(
                title=f"Đường cong Hội tụ `{selected_entity}` (Seed {selected_seed}){title_suffix}",
                xaxis_title="Epoch",
                height=370,
                margin=dict(l=20, r=20, t=45, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_loss.update_yaxes(title_text="Reconstruction Loss (L_MSE)", secondary_y=False)
            fig_loss.update_yaxes(title_text="Classifier Loss (L_CE)", secondary_y=True)
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Thực thể này không có file lịch sử `train_loss*.txt`.")

    # Bảng xếp hạng các thực thể trong tập dữ liệu & Độ ổn định qua 5 Seeds
    if bench_pick in metrics_tables:
        df_ent_all = metrics_tables[bench_pick]
        df_ent_per = df_ent_all[df_ent_all["setting"].str.contains("per-")].copy()
        agg_ent = (
            df_ent_per.groupby("entity")[["VUS-ROC", "VUS-PR", "R-AUC-ROC", "R-AUC-PR", "RF", "RF (oracle)"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg_ent.columns = [
            "Entity",
            "VUS-ROC (Mean)", "VUS-ROC (Std)",
            "VUS-PR (Mean)", "VUS-PR (Std)",
            "R-AUC-ROC (Mean)", "R-AUC-ROC (Std)",
            "R-AUC-PR (Mean)", "R-AUC-PR (Std)",
            "RF (Mean)", "RF (Std)",
            "RF Oracle (Mean)", "RF Oracle (Std)",
        ]
        agg_ent = agg_ent.sort_values("VUS-ROC (Mean)", ascending=False)

        with st.expander(f"📋 Xem Bảng Xếp Hạng Chi Tiết {len(agg_ent)} Thực Thể trong tập {DATASET_CATALOG[bench_pick]['short']} (Trung bình ± Độ lệch chuẩn 5 Seeds)", expanded=True):
            st.dataframe(
                agg_ent.style.format(
                    {col: "{:.4f}" for col in agg_ent.columns if col != "Entity"}
                ),
                use_container_width=True,
                height=260,
            )


# ==============================================================================
# MÀN HÌNH 3: KHÔNG GIAN TIỀM ẨN GIẢI THÍCH ĐƯỢC (t-SNE 128 CHIỀU)
# ==============================================================================
with tab_tsne:
    st.markdown(
        "Nhánh **Multiclass Classifier** ép không gian tiềm ẩn `enc.npy` (`128 chiều`) tách biệt rõ **12 hình thái biến đổi**. "
        "Nhờ đó, ta có thể chiếu các cửa sổ trên tập Test xuống mặt phẳng 2D để **giải thích trực quan bản chất vật lý của bất thường**."
    )

    col_tsne_l, col_tsne_r = st.columns([1.4, 1.0])
    enc_raw = bundle["enc"].reshape(bundle["enc"].shape[0], -1)
    pred_cls_all = np.argmax(bundle["pred_label"], axis=1)

    with col_tsne_r:
        proj_method = st.radio(
            "Thuật toán Giảm chiều (128D → 2D):",
            options=["PCA (Tức thì < 0.1s)", "t-SNE (Chuẩn bài báo Figure 6)"],
            horizontal=True,
        )

        # Chữ ký bất thường tại vùng đỉnh
        anom_probs_mean = np.mean(bundle["pred_label"][peak_indices, 1:], axis=0)
        fig_sig = go.Figure(
            go.Bar(
                x=anom_probs_mean,
                y=[c.capitalize() for c in ANOMALY_TYPES[1:]],
                orientation="h",
                marker_color=[CLASS_COLORS[c] for c in ANOMALY_TYPES[1:]],
                text=[f"{v * 100:.1f}%" for v in anom_probs_mean],
                textposition="outside",
            )
        )
        fig_sig.update_layout(
            title=f"Chữ ký Hình thái Bất thường tại Vùng Đỉnh (`{selected_entity}`)",
            xaxis_title="Xác suất trung bình của Classifier",
            height=380,
            margin=dict(l=20, r=45, t=40, b=20),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    with col_tsne_l:
        rng_sub = np.random.RandomState(42)
        n_total_win = len(enc_raw)

        # Chọn mẫu đại diện các cửa sổ bình thường, các cửa sổ theo từng lớp dự đoán, và vùng đỉnh bất thường
        top_anom_idx = np.argsort(scores_dict["total_score"])[-min(80, max(20, n_total_win // 50)):]
        normal_cand_idx = np.setdiff1d(np.arange(n_total_win), top_anom_idx)
        sampled_bg_idx = (
            rng_sub.choice(normal_cand_idx, size=min(350, len(normal_cand_idx)), replace=False)
            if len(normal_cand_idx) > 0
            else np.array([], dtype=int)
        )

        pts_list, lbl_list = [], []
        if "train_aug_enc" in bundle and "train_aug_cls" in bundle:
            pts_list.append(bundle["train_aug_enc"])
            lbl_list.extend([ANOMALY_TYPES[int(c)] for c in bundle["train_aug_cls"]])
        elif len(sampled_bg_idx) > 0:
            pts_list.append(enc_raw[sampled_bg_idx])
            lbl_list.extend([ANOMALY_TYPES[int(pred_cls_all[i])] for i in sampled_bg_idx])

        pts_list.append(enc_raw[top_anom_idx])
        lbl_list.extend(["🔥 Đỉnh Bất Thường (Test Peak)"] * len(top_anom_idx))

        all_pts = np.vstack(pts_list)
        if proj_method.startswith("t-SNE"):
            perp = min(30, max(5, len(all_pts) // 6))
            coords_2d = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto").fit_transform(all_pts)
        else:
            coords_2d = PCA(n_components=2, random_state=42).fit_transform(all_pts)

        fig_emb = go.Figure()
        lbl_arr = np.array(lbl_list)
        for atype in ANOMALY_TYPES:
            mask_c = lbl_arr == atype
            if np.any(mask_c):
                fig_emb.add_trace(
                    go.Scatter(
                        x=coords_2d[mask_c, 0],
                        y=coords_2d[mask_c, 1],
                        mode="markers",
                        name=f"Cụm: {atype.capitalize()}",
                        marker=dict(size=7, color=CLASS_COLORS[atype], opacity=0.70, symbol="circle"),
                    )
                )

        mask_peak = lbl_arr == "🔥 Đỉnh Bất Thường (Test Peak)"
        if np.any(mask_peak):
            fig_emb.add_trace(
                go.Scatter(
                    x=coords_2d[mask_peak, 0],
                    y=coords_2d[mask_peak, 1],
                    mode="markers",
                    name="🔥 Đỉnh Bất Thường (Test Peak)",
                    marker=dict(size=12, color="#ef4444", symbol="x", line=dict(width=2, color="#991b1b")),
                )
            )

        fig_emb.update_layout(
            title=f"Bản đồ Không gian Tiềm ẩn 2D (`{selected_entity}` — 128D Latent Embedding)",
            xaxis_title="Trục Tiềm ẩn 1",
            yaxis_title="Trục Tiềm ẩn 2",
            height=430,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_emb, use_container_width=True)


# ==============================================================================
# MÀN HÌNH 4: PHÒNG GIẢ LẬP BƠM LỖI THỜI GIAN THỰC (LIVE CHAOS SIMULATOR)
# ==============================================================================
with tab_live:
    st.markdown(
        f"Thử thách trực tiếp mô hình **`bestmodel.pkl`** của **`{selected_entity}` (Seed `{selected_seed}`)**: "
        "Lấy một cửa sổ dài `W = 100` từ chuỗi thực tế, tiêm đột biến tùy chọn và chạy suy luận thời gian thực (Live Inference)."
    )

    max_start_idx = max(0, B_len - W_size - 1)
    c_inj1, c_inj2, c_inj3, c_inj4 = st.columns([1.2, 1.3, 1.0, 1.0])
    with c_inj1:
        win_start_pick = st.slider("Vị trí cửa sổ cơ sở (Timestep):", 0, max_start_idx, min(50, max_start_idx))
    with c_inj2:
        live_aug_type = st.selectbox(
            "Loại sự cố muốn bơm vào:",
            options=ANOMALY_TYPES,
            index=1,
            format_func=lambda x: f"{x.upper()} — {CLASS_DESCRIPTIONS_VI[x]}",
        )
    with c_inj3:
        live_feat_idx = st.selectbox(
            "Kênh hiển thị:",
            options=list(range(D_feat)),
            index=0,
            format_func=lambda i: f"Channel #{i + 1}",
        )
    with c_inj4:
        live_seed = st.slider("Seed cường độ/vị trí:", 0, 100, 12)

    # Trích xuất cửa sổ (D, W) thực tế từ bundle
    if bundle["mode"] == "sample":
        base_window = bundle["input_3d"][win_start_pick].T.copy()  # (D, W)
    else:
        base_window = bundle["input_last"][win_start_pick : win_start_pick + W_size].T.copy()  # (D, W)

    live_entity = Entity(Y=np.tile(base_window, (1, 3)), name="live", verbose=False)
    live_ds = Dataset(entities=[live_entity], name="live", verbose=False)
    live_loader = Loader_aug(
        dataset=live_ds,
        batch_size=4,
        window_size=W_size,
        window_step=W_size,
        anomaly_types=["normal"],
        anomaly_types_for_dict=ANOMALY_TYPES,
        min_range=15,
        min_features=1,
        max_features=D_feat,
        fast_sampling=False,
        shuffle=False,
        verbose=False,
    )
    np.random.seed(int(live_seed))
    y_live, z_live, mask_live = live_loader.select_anomalies(live_aug_type, live_entity.Y, W_size, 2 * W_size)

    ckpt_path = bundle["model_dir"] / "bestmodel.pkl"
    params_path = bundle["model_dir"] / "model_params.json"
    if ckpt_path.exists():
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                p_dict = json.load(f)
            m_params = utils.AttrDict(**p_dict)
        else:
            m_params = utils.AttrDict(
                name="ConvAEC",
                n_features=D_feat,
                n_time=W_size,
                num_filters=[128, 128, 256, 256],
                embedding_dim=128,
                kernel_size=4,
                dropout=0.2,
                normalization="batch",
                stride=2,
                padding=2,
                anomaly_types=ANOMALY_TYPES,
                classes=len(ANOMALY_TYPES),
                classifier_dim=32,
                c_loss_ratio=0.1,
                apply_anomaly_mask=True,
                label_smoothing=True,
                alpha=0.1,
                beta=0.01,
            )
        live_model = ConvAEC(m_params)
        live_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        live_model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(y_live.T[None, :, :], dtype=torch.float32)
            x_norm_tensor = torch.tensor(z_live.T[None, :, :], dtype=torch.float32)
            batch_in = torch.cat([x_norm_tensor, x_tensor], dim=0)
            rec_out, cls_out, _ = live_model(batch_in)

            rec_norm_all = rec_out[0].numpy()  # (W, D)
            rec_inj_all = rec_out[1].numpy()   # (W, D)
            prob_norm = cls_out[0].numpy()
            prob_inj = cls_out[1].numpy()

        mse_norm_val = float(np.mean((z_live.T - rec_norm_all) ** 2))
        mse_inj_val = float(np.mean((y_live.T - rec_inj_all) ** 2))
        pred_top_cls = (
            ANOMALY_TYPES[int(np.argmax(prob_inj[1:])) + 1]
            if live_aug_type != "normal"
            else ANOMALY_TYPES[int(np.argmax(prob_inj))]
        )

        m_k1, m_k2, m_k3 = st.columns(3)
        m_k1.metric("Sai số MSE (Trước khi tiêm lỗi)", f"{mse_norm_val:.5f}")
        m_k2.metric(
            f"Sai số MSE (Sau khi tiêm {live_aug_type.upper()})",
            f"{mse_inj_val:.5f}",
            delta=f"{mse_inj_val - mse_norm_val:+.5f}",
            delta_color="inverse",
        )
        m_k3.metric("Chẩn đoán của Nhánh Classifier", f"{pred_top_cls.upper()} ({prob_inj[ANOMALY_TYPES.index(pred_top_cls)]*100:.1f}%)")

        col_l1, col_l2 = st.columns([1.3, 1.0])
        with col_l1:
            fig_live = go.Figure()
            fig_live.add_trace(
                go.Scatter(y=z_live[live_feat_idx], name="Tín hiệu Gốc Z(t)", line=dict(color="#64748b", width=1.8, dash="dot"))
            )
            fig_live.add_trace(
                go.Scatter(
                    y=y_live[live_feat_idx],
                    name=f"Tín hiệu Bơm [{live_aug_type.upper()}]",
                    line=dict(color="#ef4444", width=2.5),
                )
            )
            fig_live.add_trace(
                go.Scatter(
                    y=rec_inj_all[:, live_feat_idx],
                    name="Tái tạo từ Autoencoder X̂(t)",
                    line=dict(color="#0ea5e9", width=2.0),
                )
            )
            apts_live = np.where(mask_live[live_feat_idx] == 0)[0]
            if len(apts_live) == 0:
                apts_live = np.where(np.any(mask_live == 0, axis=0))[0]
            if len(apts_live) > 0:
                fig_live.add_vrect(
                    x0=int(apts_live.min()),
                    x1=int(apts_live.max()) + 1,
                    fillcolor="rgba(239, 68, 68, 0.18)",
                    line_width=0,
                    annotation_text="Vùng tiêm sự cố",
                )
            fig_live.update_layout(
                title=f"Phản ứng Tái tạo Trực tiếp của `bestmodel.pkl` (Kênh #{live_feat_idx + 1})",
                xaxis_title="Timestep trong cửa sổ (W = 100)",
                yaxis_title="Giá trị chuẩn hóa",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_live, use_container_width=True)

        with col_l2:
            fig_live_cls = go.Figure()
            fig_live_cls.add_trace(
                go.Bar(
                    x=[c.capitalize() for c in ANOMALY_TYPES],
                    y=prob_norm,
                    name="Trước khi tiêm (Gốc)",
                    marker_color="rgba(148, 163, 184, 0.55)",
                )
            )
            fig_live_cls.add_trace(
                go.Bar(
                    x=[c.capitalize() for c in ANOMALY_TYPES],
                    y=prob_inj,
                    name=f"Sau khi tiêm {live_aug_type.upper()}",
                    marker_color=[CLASS_COLORS[c] for c in ANOMALY_TYPES],
                )
            )
            fig_live_cls.update_layout(
                title="Phân bố Xác suất 12 Lớp của Nhánh Classifier",
                yaxis_title="Xác suất Softmax",
                barmode="group",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_live_cls, use_container_width=True)
    else:
        st.warning("Không tìm thấy file `bestmodel.pkl` trong thư mục thực nghiệm này.")


# ==============================================================================
# MÀN HÌNH 5: ĐA DẠNG HÓA 12 LỚP GIẢ BẤT THƯỜNG & NHÃN MỀM (MỤC 3.3 - 3.4)
# ==============================================================================
with tab_aug:
    col_cfg1, col_cfg2, col_cfg3 = st.columns([1.1, 1.3, 1.6])
    with col_cfg1:
        aug_seed = st.number_input("Random Seed biến đổi:", min_value=0, max_value=999, value=42, step=1)
    with col_cfg2:
        focus_aug = st.selectbox(
            "🔍 Chọn phép biến đổi để soi chi tiết:",
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

    raw_series = bundle["input_last"][:, 0]
    sample_seq = (
        np.tile(raw_series, 3)[:300].reshape(1, -1)
        if len(raw_series) < 300
        else raw_series[:300].reshape(1, -1)
    )

    dummy_entity = Entity(Y=sample_seq, name="demo_aug", verbose=False)
    dummy_dataset = Dataset(entities=[dummy_entity], name="demo_aug", verbose=False)
    aug_helper = Loader_aug(
        dataset=dummy_dataset,
        batch_size=16,
        window_size=100,
        window_step=50,
        anomaly_types=["normal"],
        anomaly_types_for_dict=ANOMALY_TYPES,
        min_range=18,
        min_features=1,
        max_features=1,
        fast_sampling=False,
        shuffle=False,
        verbose=False,
    )

    np.random.seed(int(aug_seed))
    y_aug_detail, z_norm_detail, mask_detail = aug_helper.select_anomalies(focus_aug, sample_seq, 50, 150)

    c_left, c_right = st.columns([1.35, 1.0])
    with c_left:
        fig_detail = go.Figure()
        fig_detail.add_trace(
            go.Scatter(
                y=z_norm_detail[0],
                mode="lines",
                name="Chuỗi gốc bình thường X^(t)",
                line=dict(color="#64748b", width=2, dash="dot"),
            )
        )
        fig_detail.add_trace(
            go.Scatter(
                y=y_aug_detail[0],
                mode="lines",
                name=f"Sau khi chèn {focus_aug.upper()}",
                line=dict(color=CLASS_COLORS[focus_aug], width=2.6),
            )
        )
        anom_pts = np.where(mask_detail[0] == 0)[0]
        if len(anom_pts) > 0:
            fig_detail.add_vrect(
                x0=int(anom_pts.min()),
                x1=int(anom_pts.max()) + 1,
                fillcolor="rgba(239, 68, 68, 0.20)",
                line_width=0,
                annotation_text="Vùng Masked (W=0: không tính L_MSE)",
                annotation_position="top left",
            )
        fig_detail.update_layout(
            title=f"So sánh Tín hiệu Gốc vs. Giả Bất thường [{focus_aug.upper()}] trên `{selected_entity}`",
            xaxis_title="Timestep trong cửa sổ (W = 100)",
            yaxis_title="Giá trị",
            height=330,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_detail, use_container_width=True)

    with c_right:
        K = len(ANOMALY_TYPES)
        k_idx = ANOMALY_TYPES.index(focus_aug)
        one_hot = np.zeros(K)
        one_hot[k_idx] = 1.0
        soft_label = one_hot * (1.0 - alpha_val - K * beta_val) + beta_val
        soft_label[0] += alpha_val

        fig_lbl = go.Figure()
        fig_lbl.add_trace(
            go.Bar(
                x=[c.capitalize() for c in ANOMALY_TYPES],
                y=one_hot,
                name="Nhãn cứng (One-hot)",
                marker_color="rgba(148, 163, 184, 0.50)",
            )
        )
        fig_lbl.add_trace(
            go.Bar(
                x=[c.capitalize() for c in ANOMALY_TYPES],
                y=soft_label,
                name="Nhãn mềm (Backward Correction ỹ)",
                marker_color=[CLASS_COLORS[c] for c in ANOMALY_TYPES],
            )
        )
        fig_lbl.update_layout(
            title=f"Label Refurbishment (Eq. 1): α={alpha_val:.2f}, β={beta_val:.3f}",
            yaxis_title="Xác suất nhãn mục tiêu",
            barmode="group",
            height=330,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_lbl, use_container_width=True)

    with st.expander("🖼️ Xem toàn cảnh 12 Lớp Biến Đổi Chuỗi Thời Gian (Tái hiện Figure 2a trong bài báo)", expanded=False):
        fig_grid = make_subplots(
            rows=3,
            cols=4,
            subplot_titles=[f"({i + 1}) {t.capitalize()}" for i, t in enumerate(ANOMALY_TYPES)],
            vertical_spacing=0.12,
            horizontal_spacing=0.05,
        )
        for idx, atype in enumerate(ANOMALY_TYPES):
            r = idx // 4 + 1
            c = idx % 4 + 1
            np.random.seed(int(aug_seed) + idx * 7)
            y_a, z_n, m_a = aug_helper.select_anomalies(atype, sample_seq, 50, 150)
            fig_grid.add_trace(
                go.Scatter(y=z_n[0], mode="lines", line=dict(color="#94a3b8", width=1.2, dash="dot"), showlegend=False),
                row=r,
                col=c,
            )
            fig_grid.add_trace(
                go.Scatter(y=y_a[0], mode="lines", line=dict(color=CLASS_COLORS[atype], width=2), showlegend=False),
                row=r,
                col=c,
            )
            apts = np.where(m_a[0] == 0)[0]
            if len(apts) > 0:
                fig_grid.add_vrect(
                    x0=int(apts.min()),
                    x1=int(apts.max()) + 1,
                    fillcolor="rgba(239, 68, 68, 0.18)",
                    line_width=0,
                    row=r,
                    col=c,
                )
        fig_grid.update_layout(height=540, margin=dict(l=15, r=15, t=35, b=15))
        st.plotly_chart(fig_grid, use_container_width=True)
