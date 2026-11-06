"""
Đóng gói dữ liệu thực nghiệm rút gọn từ `redlamp_retraining/` vào `demo/portable_data/`
để ứng dụng Demo có thể chạy độc lập 100% trên bất kỳ máy tính nào (kể cả khi không có thư mục 15.4 GB `redlamp_retraining/`).
"""
import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
REDLAMP_DIR = ROOT_DIR / "RedLamp"
RETRAIN_DIR = ROOT_DIR / "redlamp_retraining"
PORTABLE_DIR = DEMO_DIR / "portable_data"

SHOWCASE_MULTI_SEED = {
    "aiops": [
        "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa",
        "KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3",
        "KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0",
    ],
    "smd": [
        "machine-1-1",
        "machine-1-5",
        "machine-2-1",
        "machine-3-10",
    ],
    "msl": [
        "F-5",
        "T-4",
        "M-2",
        "C-1",
    ],
    "smap": [
        "E-6",
        "A-7",
        "D-1",
        "A-1",
    ],
}


def load_nasa_gt():
    gt_info = {}
    zip_path = REDLAMP_DIR / "dataset" / "NASA.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_entries = [n for n in z.namelist() if n.endswith("labeled_anomalies.csv")]
            if csv_entries:
                with z.open(csv_entries[0]) as f:
                    df = pd.read_csv(f)
                for _, row in df.iterrows():
                    cid = str(row["chan_id"])
                    gt_info[cid] = eval(str(row["anomaly_sequences"]))
    return gt_info


def pack_single_run(test_dir: Path, model_dir: Path, out_npz: Path, gt_seqs=None, max_points=1800, max_channels=12):
    input_last = np.load(test_dir / "input_last.npy", mmap_mode="r")
    if input_last.ndim == 1:
        B, D = len(input_last), 1
        input_2d = np.array(input_last[:, None], dtype=np.float32)
    else:
        B, D = input_last.shape
        input_2d = np.array(input_last[:, :min(D, max_channels)], dtype=np.float32)

    raw_mse = np.array(np.load(test_dir / "mse.npy", mmap_mode="r"), dtype=np.float32)
    pred_label = np.array(np.load(test_dir / "pred_label.npy", mmap_mode="r"), dtype=np.float32)
    enc_mmap = np.load(test_dir / "enc.npy", mmap_mode="r")
    enc_flat = enc_mmap.reshape(B, -1)

    mean_class_probs = np.mean(pred_label, axis=0).astype(np.float32)
    raw_ce_all = np.sum(pred_label[:, 1:], axis=1)
    quick_score = (raw_mse / (np.max(raw_mse) + 1e-8)) + raw_ce_all
    peak_idx = np.argsort(quick_score)[-max(15, B // 100):]
    peak_signature = np.mean(pred_label[peak_idx, 1:], axis=0).astype(np.float32)

    # PCA 2D coordinates
    rng = np.random.RandomState(42)
    top_k_idx = np.argsort(quick_score)[-min(60, max(20, B // 40)):]
    bg_cand = np.setdiff1d(np.arange(B), top_k_idx)
    bg_idx = rng.choice(bg_cand, size=min(240, len(bg_cand)), replace=False) if len(bg_cand) > 0 else np.array([], dtype=int)
    comb_idx = np.concatenate([bg_idx, top_k_idx])
    coords = PCA(n_components=2, random_state=42).fit_transform(np.array(enc_flat[comb_idx], dtype=np.float32))
    pred_cls_sub = np.argmax(pred_label[comb_idx], axis=1).astype(np.int16)
    is_peak_arr = np.zeros(len(comb_idx), dtype=np.uint8)
    is_peak_arr[len(bg_idx):] = 1

    # Cửa sổ 100 bước thực tế (đầy đủ D chiều) phục vụ Live Injection
    W = 100
    mid_start = min(max(0, B // 3), max(0, B - W - 1))
    if input_last.ndim == 1:
        win_100 = np.array(input_last[mid_start : mid_start + W], dtype=np.float16).reshape(W, 1)
    else:
        win_100 = np.array(input_last[mid_start : mid_start + W], dtype=np.float16)

    # Downsample chuỗi thời gian để lưu nhẹ
    stride = max(1, B // max_points)
    idx_slice = np.arange(0, B, stride, dtype=np.int32)

    # Đọc lịch sử loss
    losses_dict = {}
    for fname in ("train_loss_ae", "valid_loss_ae", "train_loss_c", "valid_loss_c", "time"):
        fpath = model_dir / f"{fname}.txt"
        if fpath.exists():
            try:
                arr = np.loadtxt(fpath)
                losses_dict[fname] = np.round(arr, 5).tolist() if arr.ndim > 0 else [float(arr)]
            except Exception:
                pass

    gt_intervals = []
    if gt_seqs:
        for seq in gt_seqs:
            s_i = max(0, int(seq[0]) - (W - 1))
            e_i = min(B, int(seq[1]) - (W - 1))
            if e_i > s_i:
                gt_intervals.append([s_i, e_i])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        n_timesteps=np.int32(B),
        n_features=np.int32(D),
        stride=np.int32(stride),
        timesteps=idx_slice,
        input_sub=input_2d[idx_slice].astype(np.float16),
        raw_mse=raw_mse[idx_slice].astype(np.float32),
        pred_label=pred_label[idx_slice].astype(np.float16),
        mean_class_probs=mean_class_probs,
        peak_signature=peak_signature,
        pca_coords=coords.astype(np.float16),
        pca_cls=pred_cls_sub,
        pca_is_peak=is_peak_arr,
        win_100=win_100,
        gt_intervals=np.array(gt_intervals, dtype=np.int32),
        losses_json=json.dumps(losses_dict),
    )


def main():
    if not RETRAIN_DIR.exists():
        print("Không tìm thấy redlamp_retraining/, bỏ qua bước đóng gói.")
        return

    # 1. Copy toàn bộ file CSV
    csv_dir = PORTABLE_DIR / "csv"
    leg_dir = csv_dir / "backup_legacy_metric_20261031"
    csv_dir.mkdir(parents=True, exist_ok=True)
    leg_dir.mkdir(parents=True, exist_ok=True)

    for csv_f in RETRAIN_DIR.glob("*.csv"):
        shutil.copy2(csv_f, csv_dir / csv_f.name)
    src_leg = RETRAIN_DIR / "backup_legacy_metric_20261031"
    if src_leg.exists():
        for csv_f in src_leg.glob("*.csv"):
            shutil.copy2(csv_f, leg_dir / csv_f.name)

    # 2. Copy 4 checkpoint bestmodel.pkl đại diện (1 cho mỗi tập dữ liệu)
    ckpt_dir = PORTABLE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sources = {
        "aiops": RETRAIN_DIR / "aiops" / "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa" / "d1_b128_w100_s10" / "0" / "bestmodel.pkl",
        "smd": RETRAIN_DIR / "smd" / "machine-1-1" / "d1_b128_w100_s10" / "0" / "bestmodel.pkl",
        "msl": RETRAIN_DIR / "msl" / "d1_b128_w100_s10" / "0" / "bestmodel.pkl",
        "smap": RETRAIN_DIR / "smap" / "d1_b128_w100_s10" / "0" / "bestmodel.pkl",
    }
    for ds_k, src_p in ckpt_sources.items():
        if src_p.exists():
            shutil.copy2(src_p, ckpt_dir / f"{ds_k}_bestmodel.pkl")

    # 3. Đóng gói các thực thể (Seed 0 cho tất cả 129 thực thể + đủ 5 Seeds cho các thực thể tiêu biểu)
    nasa_gt = load_nasa_gt()
    runs_dir = PORTABLE_DIR / "runs"
    count = 0

    for ds_key in ("aiops", "smd"):
        ds_path = RETRAIN_DIR / ds_key
        for ent_dir in sorted(ds_path.iterdir()):
            if not ent_dir.is_dir():
                continue
            ent_name = ent_dir.name
            seeds_to_pack = [0, 1, 2, 3, 4] if ent_name in SHOWCASE_MULTI_SEED[ds_key] else [0]
            for s in seeds_to_pack:
                seed_dir = ent_dir / "d1_b128_w100_s10" / str(s)
                test_dir = seed_dir / "test_all"
                if (test_dir / "input_last.npy").exists():
                    out_npz = runs_dir / ds_key / ent_name / f"seed_{s}.npz"
                    pack_single_run(test_dir, seed_dir, out_npz)
                    count += 1

    for ds_key in ("msl", "smap"):
        cfg_dir = RETRAIN_DIR / ds_key / "d1_b128_w100_s10"
        for s in range(5):
            seed_dir = cfg_dir / str(s)
            test_each = seed_dir / "test_each"
            if not test_each.exists():
                continue
            for ch_dir in sorted(test_each.iterdir()):
                if not ch_dir.is_dir():
                    continue
                ch_name = ch_dir.name
                if s > 0 and ch_name not in SHOWCASE_MULTI_SEED[ds_key]:
                    continue
                t_dir = ch_dir / "test_all"
                if (t_dir / "input_last.npy").exists():
                    out_npz = runs_dir / ds_key / ch_name / f"seed_{s}.npz"
                    pack_single_run(t_dir, seed_dir, out_npz, gt_seqs=nasa_gt.get(ch_name))
                    count += 1

    print(f"[OK] Đã đóng gói thành công {count} bộ kết quả vào {PORTABLE_DIR}")


if __name__ == "__main__":
    main()

