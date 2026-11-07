"""
FastAPI Backend cho RedLamp Web Studio.
Hỗ trợ chạy độc lập hoàn toàn từ `demo/portable_data/` (25 MB — dùng khi mang sang máy khác không có `redlamp_retraining/`)
hoặc đọc trực tiếp từ `redlamp_retraining/` (15.4 GB) khi có sẵn.

Khởi chạy:
    uvicorn demo.server:app --host 0.0.0.0 --port 8080 --reload
"""
import os
import sys
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from sklearn.decomposition import PCA

DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
REDLAMP_DIR = ROOT_DIR / "RedLamp" if (ROOT_DIR / "RedLamp").exists() else ROOT_DIR / "REDLAMP"
RETRAIN_DIR = ROOT_DIR / "redlamp_retraining"
PORTABLE_DIR = DEMO_DIR / "portable_data"
STATIC_DIR = DEMO_DIR / "static"

for p in (REDLAMP_DIR, ROOT_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import utils
from models.meta import ConvAEC
from loaders.dataset import Entity, Dataset
from loaders.loader_aug import Loader_aug
from main import mse
from demo.generate_sample_data import ensure_sample_data, ANOMALY_TYPES

app = FastAPI(title="RedLamp Web Studio API", version="2.1")

DATASET_CATALOG = {
    "aiops": {
        "title": "AIOps (29 KPIs)",
        "short": "AIOps",
        "summary_csv": "summary_iops.csv",
        "metrics_csv": "metrics_iops.csv",
        "domain_badge": "Web KPI (D=1)",
        "default_entity": "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa",
    },
    "smd": {
        "title": "SMD (28 Servers)",
        "short": "SMD",
        "summary_csv": "summary_smd.csv",
        "metrics_csv": "metrics_smd.csv",
        "domain_badge": "Server (D=38)",
        "default_entity": "machine-1-1",
    },
    "msl": {
        "title": "NASA MSL (23 Kênh)",
        "short": "MSL",
        "summary_csv": "summary_msl.csv",
        "metrics_csv": "metrics_msl.csv",
        "domain_badge": "Mars Rover (D=55)",
        "default_entity": "F-5",
    },
    "smap": {
        "title": "NASA SMAP (49 Kênh)",
        "short": "SMAP",
        "summary_csv": "summary_smap.csv",
        "metrics_csv": "metrics_smap.csv",
        "domain_badge": "Satellite (D=25)",
        "default_entity": "E-6",
    },
    "sample": {
        "title": "UCR Demo (#028 & #040)",
        "short": "UCR Demo",
        "summary_csv": None,
        "metrics_csv": None,
        "domain_badge": "UCR Showcase",
        "default_entity": "028_UCR_ABP_Demo",
    },
}

_NASA_GT_CACHE = None
_SUMMARY_CACHE = None
_METRICS_CACHE = None
_CATALOG_CACHE = None


def get_csv_base_dir() -> Path:
    """Ưu tiên đọc CSV từ demo/portable_data/csv (luôn có sẵn trên mọi máy), fallback sang redlamp_retraining."""
    if (PORTABLE_DIR / "csv").exists():
        return PORTABLE_DIR / "csv"
    return RETRAIN_DIR


def get_nasa_gt():
    global _NASA_GT_CACHE
    if _NASA_GT_CACHE is not None:
        return _NASA_GT_CACHE
    gt_info = {}
    zip_path = REDLAMP_DIR / "dataset" / "NASA.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_entries = [n for n in z.namelist() if n.endswith("labeled_anomalies.csv")]
                if csv_entries:
                    with z.open(csv_entries[0]) as f:
                        df = pd.read_csv(f)
                    for _, row in df.iterrows():
                        cid = str(row["chan_id"])
                        gt_info[cid] = {
                            "sequences": eval(str(row["anomaly_sequences"])),
                            "classes": eval(str(row["class"])) if "class" in row and pd.notna(row["class"]) else [],
                            "num_values": int(row["num_values"]),
                        }
        except Exception:
            pass
    _NASA_GT_CACHE = gt_info
    return gt_info


def get_summaries_and_metrics():
    global _SUMMARY_CACHE, _METRICS_CACHE
    if _SUMMARY_CACHE is not None and _METRICS_CACHE is not None:
        return _SUMMARY_CACHE, _METRICS_CACHE

    summaries = {}
    metrics_map = {}
    csv_root = get_csv_base_dir()
    if csv_root.exists():
        for key, info in DATASET_CATALOG.items():
            s_csv = info.get("summary_csv")
            m_csv = info.get("metrics_csv")
            if s_csv and (csv_root / s_csv).exists():
                df_new = pd.read_csv(csv_root / s_csv).replace({np.nan: None})
                p_old = csv_root / "backup_legacy_metric_20261031" / s_csv
                df_old = pd.read_csv(p_old).replace({np.nan: None}) if p_old.exists() else None
                summaries[key] = {
                    "current": df_new.to_dict(orient="records"),
                    "legacy": df_old.to_dict(orient="records") if df_old is not None else [],
                }
            if m_csv and (csv_root / m_csv).exists():
                df_m = pd.read_csv(csv_root / m_csv).replace({np.nan: None})
                metrics_map[key] = df_m

    _SUMMARY_CACHE = summaries
    _METRICS_CACHE = metrics_map
    return summaries, metrics_map


def scan_all_entities():
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    ensure_sample_data()
    _, metrics_map = get_summaries_and_metrics()
    catalog = {}

    for ds_key, ds_meta in DATASET_CATALOG.items():
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
        else:
            # 1. Quét từ demo/portable_data/runs/<ds_key> trước (đảm bảo chạy trên mọi máy)
            port_ds_dir = PORTABLE_DIR / "runs" / ds_key
            if port_ds_dir.exists():
                for ent_dir in sorted(port_ds_dir.iterdir()):
                    if not ent_dir.is_dir():
                        continue
                    for npz_f in sorted(ent_dir.glob("seed_*.npz")):
                        seed_id = int(npz_f.stem.split("_")[1])
                        entity_map.setdefault(ent_dir.name, {})[seed_id] = {
                            "npz_path": str(npz_f),
                            "ckpt_path": str(PORTABLE_DIR / "checkpoints" / f"{ds_key}_bestmodel.pkl"),
                            "mode": "portable",
                        }

            # 2. Nếu máy hiện tại có thêm redlamp_retraining/, bổ sung các seed chưa có trong portable
            if RETRAIN_DIR.exists():
                ds_dir = RETRAIN_DIR / ds_key
                if ds_dir.exists():
                    if ds_key in ("aiops", "smd"):
                        for ent_dir in sorted(ds_dir.iterdir()):
                            if not ent_dir.is_dir():
                                continue
                            cfg_dir = ent_dir / "d1_b128_w100_s10"
                            if not cfg_dir.exists():
                                continue
                            for seed_dir in sorted(cfg_dir.iterdir()):
                                if seed_dir.is_dir() and seed_dir.name.isdigit():
                                    s_id = int(seed_dir.name)
                                    test_dir = seed_dir / "test_all"
                                    if (test_dir / "input_last.npy").exists():
                                        entity_map.setdefault(ent_dir.name, {}).setdefault(
                                            s_id,
                                            {
                                                "test_dir": str(test_dir),
                                                "model_dir": str(seed_dir),
                                                "mode": "retrain",
                                            },
                                        )
                    elif ds_key in ("msl", "smap"):
                        cfg_dir = ds_dir / "d1_b128_w100_s10"
                        if cfg_dir.exists():
                            for seed_dir in sorted(cfg_dir.iterdir()):
                                if not (seed_dir.is_dir() and seed_dir.name.isdigit()):
                                    continue
                                s_id = int(seed_dir.name)
                                test_each_dir = seed_dir / "test_each"
                                if test_each_dir.exists():
                                    for ch_dir in sorted(test_each_dir.iterdir()):
                                        t_dir = ch_dir / "test_all"
                                        if (t_dir / "input_last.npy").exists():
                                            entity_map.setdefault(ch_dir.name, {}).setdefault(
                                                s_id,
                                                {
                                                    "test_dir": str(t_dir),
                                                    "model_dir": str(seed_dir),
                                                    "mode": "retrain",
                                                },
                                            )

        ent_scores = {}
        if ds_key in metrics_map:
            df_m = metrics_map[ds_key]
            for ent_n, grp in df_m.groupby("entity"):
                v_roc = grp["VUS-ROC"].dropna().mean()
                r_pr = grp["R-AUC-PR"].dropna().mean()
                ent_scores[str(ent_n)] = {
                    "vus_roc": round(float(v_roc), 4) if pd.notna(v_roc) else None,
                    "rauc_pr": round(float(r_pr), 4) if pd.notna(r_pr) else None,
                }

        catalog[ds_key] = {
            **ds_meta,
            "entities": entity_map,
            "entity_scores": ent_scores,
        }

    _CATALOG_CACHE = catalog
    return catalog


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/init")
def api_init():
    catalog = scan_all_entities()
    summaries, metrics_map = get_summaries_and_metrics()

    datasets_out = {}
    leaderboards = {}
    for ds_key, info in catalog.items():
        if not info["entities"]:
            continue
        ent_list = []
        for ent_name, seeds_dict in info["entities"].items():
            sc = info["entity_scores"].get(ent_name, {})
            ent_list.append({
                "name": ent_name,
                "seeds": sorted(list(seeds_dict.keys())),
                "vus_roc": sc.get("vus_roc"),
                "rauc_pr": sc.get("rauc_pr"),
            })
        datasets_out[ds_key] = {
            "title": info["title"],
            "short": info["short"],
            "domain_badge": info["domain_badge"],
            "default_entity": info["default_entity"] if info["default_entity"] in info["entities"] else ent_list[0]["name"],
            "entities": ent_list,
        }

        if ds_key in metrics_map:
            df_m = metrics_map[ds_key]
            df_per = df_m[df_m["setting"].astype(str).str.contains("per-")].copy()
            rows = []
            for ent_n, grp in df_per.groupby("entity"):
                rows.append({
                    "entity": str(ent_n),
                    "vus_roc_mean": round(float(grp["VUS-ROC"].mean()), 4),
                    "vus_roc_std": round(float(grp["VUS-ROC"].std()), 4),
                    "vus_pr_mean": round(float(grp["VUS-PR"].mean()), 4),
                    "rauc_roc_mean": round(float(grp["R-AUC-ROC"].mean()), 4),
                    "rauc_pr_mean": round(float(grp["R-AUC-PR"].mean()), 4),
                    "rf_mean": round(float(grp["RF"].mean()), 4),
                    "rf_oracle_mean": round(float(grp["RF (oracle)"].mean()), 4) if "RF (oracle)" in grp else None,
                })
            rows.sort(key=lambda r: r["vus_roc_mean"], reverse=True)
            leaderboards[ds_key] = rows

    return JSONResponse({
        "datasets": datasets_out,
        "summaries": summaries,
        "leaderboards": leaderboards,
        "anomaly_types": ANOMALY_TYPES,
        "portable_ready": PORTABLE_DIR.exists(),
    })


@app.get("/api/run")
def api_get_run(
    dataset: str = Query(...),
    entity: str = Query(...),
    seed: int = Query(0),
    channel: int = Query(0),
    max_points: int = Query(2400),
):
    catalog = scan_all_entities()
    if dataset not in catalog or entity not in catalog[dataset]["entities"]:
        return JSONResponse({"error": "Không tìm thấy thực thể"}, status_code=404)

    seeds_map = catalog[dataset]["entities"][entity]
    if seed not in seeds_map:
        seed = sorted(seeds_map.keys())[0]
    run_info = seeds_map[seed]
    mode = run_info["mode"]
    W = 100

    _, metrics_map = get_summaries_and_metrics()
    run_metrics = None
    if dataset in metrics_map:
        df_m = metrics_map[dataset]
        lookup_ent = dataset if entity.startswith("[Aggregate]") else entity
        sub_m = df_m[(df_m["entity"].astype(str) == str(lookup_ent)) & (df_m["seed"] == int(seed))]
        if len(sub_m) > 0:
            row = sub_m.iloc[0]
            run_metrics = {
                "vus_roc": round(float(row["VUS-ROC"]), 4) if pd.notna(row["VUS-ROC"]) else None,
                "vus_pr": round(float(row["VUS-PR"]), 4) if pd.notna(row["VUS-PR"]) else None,
                "rauc_roc": round(float(row["R-AUC-ROC"]), 4) if pd.notna(row["R-AUC-ROC"]) else None,
                "rauc_pr": round(float(row["R-AUC-PR"]), 4) if pd.notna(row["R-AUC-PR"]) else None,
                "rf": round(float(row["RF"]), 4) if pd.notna(row["RF"]) else None,
                "rf_oracle": round(float(row["RF (oracle)"]), 4) if "RF (oracle)" in row and pd.notna(row["RF (oracle)"]) else None,
            }

    if mode == "portable":
        # Đọc siêu tốc (< 3ms) từ gói nén demo/portable_data/runs/...
        z = np.load(run_info["npz_path"], allow_pickle=False)
        B = int(z["n_timesteps"])
        D = int(z["n_features"])
        stride = int(z["stride"])
        timesteps = z["timesteps"].astype(int).tolist()
        input_sub = z["input_sub"].astype(np.float32)
        avail_ch = input_sub.shape[1]
        ch = max(0, min(avail_ch - 1, int(channel)))
        input_1d = np.round(input_sub[:, ch], 5).tolist()
        raw_mse = np.round(z["raw_mse"].astype(np.float32), 6).tolist()
        pred_label = np.round(z["pred_label"].astype(np.float32), 4).tolist()
        mean_class_probs = np.round(z["mean_class_probs"].astype(np.float32), 5).tolist()
        peak_signature = np.round(z["peak_signature"].astype(np.float32), 4).tolist()
        gt_intervals = z["gt_intervals"].tolist() if "gt_intervals" in z and len(z["gt_intervals"]) > 0 else []
        losses = json.loads(str(z["losses_json"])) if "losses_json" in z else {}

        coords = z["pca_coords"].astype(np.float32)
        p_cls = z["pca_cls"].astype(int)
        p_peak = z["pca_is_peak"].astype(int)
        emb_points = []
        for i_pt in range(len(coords)):
            is_pk = bool(p_peak[i_pt] == 1)
            emb_points.append({
                "x": round(float(coords[i_pt, 0]), 4),
                "y": round(float(coords[i_pt, 1]), 4),
                "group": "peak" if is_pk else ANOMALY_TYPES[int(p_cls[i_pt])],
                "is_peak": is_pk,
            })

        return JSONResponse({
            "dataset": dataset,
            "entity": entity,
            "seed": seed,
            "n_timesteps": B,
            "n_features": avail_ch,
            "total_features": D,
            "channel": ch,
            "stride": stride,
            "window_size": W,
            "timesteps": timesteps,
            "input_series": input_1d,
            "pred_series": None,
            "raw_mse": raw_mse,
            "pred_label": pred_label,
            "mean_class_probs": mean_class_probs,
            "peak_signature": peak_signature,
            "gt_intervals": gt_intervals,
            "losses": losses,
            "metrics": run_metrics,
            "embedding_2d": emb_points,
        })

    # Chế độ đọc từ sample hoặc redlamp_retraining
    test_dir = Path(run_info["test_dir"])
    model_dir = Path(run_info["model_dir"])
    gt_intervals = []

    if mode == "sample":
        inputs = np.load(test_dir / "input.npy", mmap_mode="r")
        preds = np.load(test_dir / "pred.npy", mmap_mode="r")
        B, W, D = inputs.shape
        ch = max(0, min(D - 1, int(channel)))
        input_1d = np.array(inputs[:, -1, ch], dtype=np.float32)
        pred_1d = np.array(preds[:, -1, ch], dtype=np.float32)
        raw_mse = np.array(mse(inputs.reshape(B, -1), preds.reshape(B, -1)), dtype=np.float32)
        pred_label = np.array(np.load(test_dir / "pred_label.npy", mmap_mode="r"), dtype=np.float32)
        if (test_dir / "meta.json").exists():
            with open(test_dir / "meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            st_i = max(0, int(meta.get("anomaly_start_in_test", 0)) - (W - 1))
            ed_i = max(0, int(meta.get("anomaly_end_in_test", 0)) - (W - 1))
            if ed_i > st_i:
                gt_intervals.append([st_i, ed_i])
    else:
        input_last = np.load(test_dir / "input_last.npy", mmap_mode="r")
        if input_last.ndim == 1:
            B, D = len(input_last), 1
            ch = 0
            input_1d = np.array(input_last, dtype=np.float32)
        else:
            B, D = input_last.shape
            ch = max(0, min(D - 1, int(channel)))
            input_1d = np.array(input_last[:, ch], dtype=np.float32)
        pred_1d = None
        raw_mse = np.array(np.load(test_dir / "mse.npy", mmap_mode="r"), dtype=np.float32)
        pred_label = np.array(np.load(test_dir / "pred_label.npy", mmap_mode="r"), dtype=np.float32)

        nasa_gt = get_nasa_gt()
        if entity in nasa_gt:
            for seq in nasa_gt[entity]["sequences"]:
                s_i = max(0, int(seq[0]) - (W - 1))
                e_i = min(B, int(seq[1]) - (W - 1))
                if e_i > s_i:
                    gt_intervals.append([s_i, e_i])

    mean_class_probs = np.mean(pred_label, axis=0).astype(float).tolist()
    losses = {}
    for fname in ("train_loss_ae", "valid_loss_ae", "train_loss_c", "valid_loss_c", "time"):
        fpath = model_dir / f"{fname}.txt"
        if fpath.exists():
            try:
                arr = np.loadtxt(fpath)
                losses[fname] = np.round(arr, 5).tolist() if arr.ndim > 0 else [float(arr)]
            except Exception:
                pass

    raw_ce_all = np.sum(pred_label[:, 1:], axis=1)
    quick_score = (raw_mse / (np.max(raw_mse) + 1e-8)) + raw_ce_all
    peak_idx = np.argsort(quick_score)[-max(15, B // 100):]
    peak_signature = np.mean(pred_label[peak_idx, 1:], axis=0).astype(float).tolist()

    stride = max(1, B // max_points)
    idx_slice = np.arange(0, B, stride)

    enc_mmap = np.load(test_dir / "enc.npy", mmap_mode="r")
    enc_flat = enc_mmap.reshape(B, -1)
    rng = np.random.RandomState(42)
    top_k_idx = np.argsort(quick_score)[-min(65, max(20, B // 40)):]
    bg_cand = np.setdiff1d(np.arange(B), top_k_idx)
    bg_idx = rng.choice(bg_cand, size=min(280, len(bg_cand)), replace=False) if len(bg_cand) > 0 else np.array([], dtype=int)

    emb_points = []
    if mode == "sample" and (test_dir / "train_aug_enc.npy").exists():
        aug_enc = np.load(test_dir / "train_aug_enc.npy")
        aug_cls = np.load(test_dir / "train_aug_cls.npy")
        all_mat = np.vstack([aug_enc, enc_flat[top_k_idx]])
        coords = PCA(n_components=2, random_state=42).fit_transform(all_mat)
        for i_pt, c_id in enumerate(aug_cls):
            emb_points.append({
                "x": round(float(coords[i_pt, 0]), 4),
                "y": round(float(coords[i_pt, 1]), 4),
                "group": ANOMALY_TYPES[int(c_id)],
                "is_peak": False,
            })
        offset = len(aug_cls)
        for j_pt in range(len(top_k_idx)):
            emb_points.append({
                "x": round(float(coords[offset + j_pt, 0]), 4),
                "y": round(float(coords[offset + j_pt, 1]), 4),
                "group": "peak",
                "is_peak": True,
            })
    else:
        comb_idx = np.concatenate([bg_idx, top_k_idx])
        sub_enc = np.array(enc_flat[comb_idx], dtype=np.float32)
        coords = PCA(n_components=2, random_state=42).fit_transform(sub_enc)
        pred_cls_sub = np.argmax(pred_label[comb_idx], axis=1)
        for i_pt in range(len(bg_idx)):
            emb_points.append({
                "x": round(float(coords[i_pt, 0]), 4),
                "y": round(float(coords[i_pt, 1]), 4),
                "group": ANOMALY_TYPES[int(pred_cls_sub[i_pt])],
                "is_peak": False,
            })
        offset = len(bg_idx)
        for j_pt in range(len(top_k_idx)):
            emb_points.append({
                "x": round(float(coords[offset + j_pt, 0]), 4),
                "y": round(float(coords[offset + j_pt, 1]), 4),
                "group": "peak",
                "is_peak": True,
            })

    return JSONResponse({
        "dataset": dataset,
        "entity": entity,
        "seed": seed,
        "n_timesteps": int(B),
        "n_features": int(D),
        "total_features": int(D),
        "channel": int(ch),
        "stride": int(stride),
        "window_size": int(W),
        "timesteps": idx_slice.tolist(),
        "input_series": np.round(input_1d[idx_slice], 5).tolist(),
        "pred_series": np.round(pred_1d[idx_slice], 5).tolist() if pred_1d is not None else None,
        "raw_mse": np.round(raw_mse[idx_slice], 6).tolist(),
        "pred_label": np.round(pred_label[idx_slice], 4).tolist(),
        "mean_class_probs": np.round(mean_class_probs, 5).tolist(),
        "peak_signature": np.round(peak_signature, 4).tolist(),
        "gt_intervals": gt_intervals,
        "losses": losses,
        "metrics": run_metrics,
        "embedding_2d": emb_points,
    })


@app.get("/api/inject")
def api_live_inject(
    dataset: str = Query(...),
    entity: str = Query(...),
    seed: int = Query(0),
    win_start: int = Query(50),
    aug_type: str = Query("spike"),
    channel: int = Query(0),
    inj_seed: int = Query(12),
):
    catalog = scan_all_entities()
    if dataset not in catalog or entity not in catalog[dataset]["entities"]:
        return JSONResponse({"error": "Không tìm thấy thực thể"}, status_code=404)

    seeds_map = catalog[dataset]["entities"][entity]
    if seed not in seeds_map:
        seed = sorted(seeds_map.keys())[0]
    run_info = seeds_map[seed]
    mode = run_info["mode"]
    W = 100

    if mode == "portable":
        z = np.load(run_info["npz_path"], allow_pickle=False)
        win_100 = z["win_100"].astype(np.float32)  # (W, D)
        base_window = win_100.T.copy()  # (D, W)
        D = base_window.shape[0]
        ckpt_path = Path(run_info["ckpt_path"])
        params_path = Path("")
    elif mode == "sample":
        test_dir = Path(run_info["test_dir"])
        model_dir = Path(run_info["model_dir"])
        inputs = np.load(test_dir / "input.npy", mmap_mode="r")
        B, W, D = inputs.shape
        idx = max(0, min(B - 1, int(win_start)))
        base_window = np.array(inputs[idx].T, dtype=np.float32)
        ckpt_path = model_dir / "bestmodel.pkl"
        params_path = model_dir / "model_params.json"
    else:
        test_dir = Path(run_info["test_dir"])
        model_dir = Path(run_info["model_dir"])
        input_last = np.load(test_dir / "input_last.npy", mmap_mode="r")
        if input_last.ndim == 1:
            B, D = len(input_last), 1
            idx = max(0, min(B - W - 1, int(win_start)))
            base_window = np.array(input_last[idx : idx + W], dtype=np.float32).reshape(1, W)
        else:
            B, D = input_last.shape
            idx = max(0, min(B - W - 1, int(win_start)))
            base_window = np.array(input_last[idx : idx + W].T, dtype=np.float32)
        ckpt_path = model_dir / "bestmodel.pkl"
        params_path = model_dir / "model_params.json"

    ch = max(0, min(D - 1, int(channel)))
    live_entity = Entity(Y=np.tile(base_window, (1, 3)), name="live", verbose=False)
    live_ds = Dataset(entities=[live_entity], name="live", verbose=False)
    live_loader = Loader_aug(
        dataset=live_ds,
        batch_size=4,
        window_size=W,
        window_step=W,
        anomaly_types=["normal"],
        anomaly_types_for_dict=ANOMALY_TYPES,
        min_range=15,
        min_features=1,
        max_features=D,
        fast_sampling=False,
        shuffle=False,
        verbose=False,
    )
    np.random.seed(int(inj_seed))
    if aug_type not in ANOMALY_TYPES:
        aug_type = "spike"
    y_live, z_live, mask_live = live_loader.select_anomalies(aug_type, live_entity.Y, W, 2 * W)

    grid_12 = {}
    single_seq = np.tile(base_window[ch : ch + 1, :], (1, 3))
    e_1d = Entity(Y=single_seq, name="g12", verbose=False)
    ds_1d = Dataset(entities=[e_1d], name="g12", verbose=False)
    loader_1d = Loader_aug(
        dataset=ds_1d, batch_size=4, window_size=W, window_step=W,
        anomaly_types=["normal"], anomaly_types_for_dict=ANOMALY_TYPES,
        min_range=15, min_features=1, max_features=1,
        fast_sampling=False, shuffle=False, verbose=False,
    )
    for idx_a, atype in enumerate(ANOMALY_TYPES):
        np.random.seed(int(inj_seed) + idx_a * 7)
        ya, zn, ma = loader_1d.select_anomalies(atype, single_seq, W, 2 * W)
        apts = np.where(ma[0] == 0)[0]
        grid_12[atype] = {
            "orig": np.round(zn[0], 4).tolist(),
            "aug": np.round(ya[0], 4).tolist(),
            "mask_range": [int(apts.min()), int(apts.max())] if len(apts) > 0 else None,
        }

    if not ckpt_path.exists():
        return JSONResponse({"error": "Không tìm thấy bestmodel.pkl"}, status_code=404)

    if params_path.is_file():
        with open(params_path, "r", encoding="utf-8") as f:
            p_dict = json.load(f)
        m_params = utils.AttrDict(**p_dict)
    else:
        m_params = utils.AttrDict(
            name="ConvAEC", n_features=D, n_time=W,
            num_filters=[128, 128, 256, 256], embedding_dim=128,
            kernel_size=4, dropout=0.2, normalization="batch",
            stride=2, padding=2, anomaly_types=ANOMALY_TYPES,
            classes=len(ANOMALY_TYPES), classifier_dim=32,
            c_loss_ratio=0.1, apply_anomaly_mask=True,
            label_smoothing=True, alpha=0.1, beta=0.01,
        )

    live_model = ConvAEC(m_params)
    live_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    live_model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(y_live.T[None, :, :], dtype=torch.float32)
        x_norm_tensor = torch.tensor(z_live.T[None, :, :], dtype=torch.float32)
        batch_in = torch.cat([x_norm_tensor, x_tensor], dim=0)
        rec_out, cls_out, _ = live_model(batch_in)
        rec_norm_all = rec_out[0].numpy()
        rec_inj_all = rec_out[1].numpy()
        prob_norm = cls_out[0].numpy()
        prob_inj = cls_out[1].numpy()

    mse_norm_val = float(np.mean((z_live.T - rec_norm_all) ** 2))
    mse_inj_val = float(np.mean((y_live.T - rec_inj_all) ** 2))
    apts_ch = np.where(mask_live[ch] == 0)[0]
    if len(apts_ch) == 0:
        apts_ch = np.where(np.any(mask_live == 0, axis=0))[0]

    return JSONResponse({
        "aug_type": aug_type,
        "channel": ch,
        "orig_signal": np.round(z_live[ch], 4).tolist(),
        "inj_signal": np.round(y_live[ch], 4).tolist(),
        "rec_signal": np.round(rec_inj_all[:, ch], 4).tolist(),
        "mask_range": [int(apts_ch.min()), int(apts_ch.max())] if len(apts_ch) > 0 else None,
        "mse_before": round(mse_norm_val, 6),
        "mse_after": round(mse_inj_val, 6),
        "prob_before": np.round(prob_norm, 4).tolist(),
        "prob_after": np.round(prob_inj, 4).tolist(),
        "grid_12": grid_12,
    })
