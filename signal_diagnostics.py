import os
os.environ['OMP_NUM_THREADS'] = '8'

from datetime import datetime
import numpy as np

from dataset_robust import FlexibleBloodFlowDataset


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remain = seconds % 60
    return f"{hours}h {minutes}m {remain:.1f}s"


def compute_scalar_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_true - y_pred)
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(abs_err / np.maximum(np.abs(y_true), 1e-8)) * 100.0
    return mae, rmse, mape


def _compute_temporal_features(counts):
    """Compute μs-level temporal features from per-frame event counts (length ~5000)."""
    n = len(counts)
    mean_c = counts.mean()
    feats = {}

    # --- autocorrelation at multiple lags ---
    c_norm = counts - mean_c
    denom = np.dot(c_norm, c_norm)

    def _acf(lag):
        if denom < 1e-10:
            return 0.0
        return np.dot(c_norm[lag:], c_norm[:n - lag]) / denom

    feats["autocorr_lag1"] = _acf(1)
    feats["autocorr_lag5"] = _acf(5)
    feats["autocorr_lag10"] = _acf(10)
    feats["autocorr_lag20"] = _acf(20)
    feats["autocorr_lag50"] = _acf(50)

    # exponential decay: ACF(k) ≈ exp(-k/τ) → τ = -k / log(ACF)
    lags = np.array([1, 5, 10, 20, 50], dtype=np.float64)
    acf_vals = np.array(
        [feats["autocorr_lag1"], feats["autocorr_lag5"],
         feats["autocorr_lag10"], feats["autocorr_lag20"],
         feats["autocorr_lag50"]]
    )
    valid = acf_vals > 1e-6
    if valid.sum() >= 3:
        log_acf = np.log(np.maximum(acf_vals[valid], 1e-10))
        slope = np.linalg.lstsq(lags[valid].reshape(-1, 1), log_acf, rcond=None)[0][0]
        feats["autocorr_decay_tau"] = float(np.clip(-1.0 / slope, 1.0, 50000.0)) if slope < 0 else 50000.0
    else:
        feats["autocorr_decay_tau"] = 50000.0

    # --- frame-difference statistics (μs-level volatility) ---
    diffs = np.abs(np.diff(counts))
    feats["diff_mean"] = diffs.mean()
    feats["diff_std"] = diffs.std()
    feats["diff_max"] = diffs.max()
    feats["diff_cv"] = float(diffs.std() / max(diffs.mean(), 1e-8))

    # --- multi-scale Fano factor (variance / mean) ---
    feats["fano_20us"] = float(counts.var() / max(mean_c, 1e-8))

    if n >= 50:
        bins_1ms = counts[:n - n % 50].reshape(-1, 50).sum(axis=1)
        feats["fano_1ms"] = float(bins_1ms.var() / max(bins_1ms.mean(), 1e-8))
    else:
        feats["fano_1ms"] = 0.0

    if n >= 500:
        bins_10ms = counts[:n - n % 500].reshape(-1, 500).sum(axis=1)
        feats["fano_10ms"] = float(bins_10ms.var() / max(bins_10ms.mean(), 1e-8))
    else:
        feats["fano_10ms"] = 0.0

    # --- spectral features ---
    fft = np.abs(np.fft.rfft(counts))
    freqs = np.fft.rfftfreq(n)
    fft_no_dc = fft[1:]
    freqs_no_dc = freqs[1:]
    total_energy = np.sum(fft_no_dc)
    if total_energy > 1e-10:
        feats["spectral_centroid"] = float(np.sum(freqs_no_dc * fft_no_dc) / total_energy)
        cumsum = np.cumsum(fft_no_dc)
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * total_energy))
        feats["spectral_rolloff"] = float(freqs_no_dc[min(rolloff_idx, len(freqs_no_dc) - 1)])
        feats["high_freq_ratio"] = float(np.sum(fft_no_dc[len(fft_no_dc) // 2:]) / total_energy)
    else:
        feats["spectral_centroid"] = 0.0
        feats["spectral_rolloff"] = 0.0
        feats["high_freq_ratio"] = 0.0

    # --- zero-crossing rate (temporal roughness) ---
    detrended = counts - mean_c
    feats["zero_crossing_rate"] = float(np.sum(np.diff(np.signbit(detrended))) / max(n - 1, 1))

    # --- burst index: fraction of 50-frame windows exceeding 2× global mean ---
    if n >= 50:
        win_cnt = n // 50
        window_means = counts[:win_cnt * 50].reshape(win_cnt, 50).mean(axis=1)
        feats["burst_index"] = float(np.mean(window_means > 2 * mean_c))
    else:
        feats["burst_index"] = 0.0

    # --- event accumulation ratio (second half / first half) ---
    half = n // 2
    first = counts[:half].sum()
    second = counts[half:].sum()
    feats["event_growth_ratio"] = float(second / max(first, 1.0))

    return feats


def extract_sample_features(sample):
    sequence_data, velocity, d_value = sample
    counts = np.fromiter((coords.shape[0] for coords, _ in sequence_data), dtype=np.float64)

    # === kept from original features ===
    total_events = counts.sum()

    # === commented-out low-discrimination features ===
    # active_frames = np.count_nonzero(counts)                  # constant ~4975-4990
    # active_frame_ratio = active_frames / max(len(counts), 1)   # constant ~1.0
    # events_per_frame_mean = counts.mean()                      # = total_events/5000, collinear
    # events_per_frame_std = counts.std()                        # collinear with total
    # events_per_frame_max = counts.max()                        # collinear with total
    #
    # frame_indices = np.arange(len(counts), dtype=np.float64)
    # center_time = float((counts * frame_indices).sum() / total_events) if total_events > 0 else 0.0
    # event_center_time_norm = center_time / max(len(counts) - 1, 1)  # constant ~0.50
    #
    # first_active = int(np.argmax(counts > 0)) if total_events > 0 else -1
    # first_active_time_norm = first_active / max(len(counts) - 1, 1) if first_active >= 0 else 0.0  # constant ~0
    #
    # last_active = int(len(counts) - 1 - np.argmax(counts[::-1] > 0)) if total_events > 0 else -1
    # last_active_time_norm = last_active / max(len(counts) - 1, 1) if last_active >= 0 else 0.0  # constant ~1
    #
    # quartiles = np.array_split(counts, 4)
    # quartile_sums  = [part.sum() for part in quartiles]   # all ~total_events/4
    # quartile_means = [part.mean() for part in quartiles]   # all ~events_per_frame_mean

    # === new μs-level temporal features ===
    tf = _compute_temporal_features(counts)

    features = np.array(
        [
            total_events,                 # 0
            float(d_value),               # 1
            tf["autocorr_lag1"],          # 2
            tf["autocorr_lag5"],          # 3
            tf["autocorr_lag10"],         # 4
            tf["autocorr_lag20"],         # 5
            tf["autocorr_lag50"],         # 6
            tf["autocorr_decay_tau"],     # 7
            tf["diff_mean"],              # 8
            tf["diff_std"],               # 9
            tf["diff_max"],               # 10
            tf["diff_cv"],                # 11
            tf["fano_20us"],              # 12
            tf["fano_1ms"],               # 13
            tf["fano_10ms"],              # 14
            tf["spectral_centroid"],      # 15
            tf["spectral_rolloff"],       # 16
            tf["high_freq_ratio"],        # 17
            tf["zero_crossing_rate"],     # 18
            tf["burst_index"],            # 19
            tf["event_growth_ratio"],     # 20
        ],
        dtype=np.float64,
    )
    return features, float(velocity), float(d_value)


FEATURE_NAMES = [
    "total_events",           # 0
    "d_value",                # 1
    "autocorr_lag1",          # 2
    "autocorr_lag5",          # 3
    "autocorr_lag10",         # 4
    "autocorr_lag20",         # 5
    "autocorr_lag50",         # 6
    "autocorr_decay_tau",     # 7
    "diff_mean",              # 8
    "diff_std",               # 9
    "diff_max",               # 10
    "diff_cv",                # 11
    "fano_20us",              # 12
    "fano_1ms",               # 13
    "fano_10ms",              # 14
    "spectral_centroid",      # 15
    "spectral_rolloff",       # 16
    "high_freq_ratio",        # 17
    "zero_crossing_rate",     # 18
    "burst_index",            # 19
    "event_growth_ratio",     # 20
]

# indices into the feature vector for per-velocity summary table
_IDX_TOTAL_EVENTS = 0
_IDX_AUTOCORR_LAG1 = 2
_IDX_FANO_20US = 12
_IDX_DIFF_CV = 11
_IDX_DIFF_MEAN = 8
_IDX_DIFF_STD = 9
_IDX_DIFF_MAX = 10

ENV_NORMALIZE_FEATURE_INDICES = [
    _IDX_TOTAL_EVENTS,
    _IDX_DIFF_MEAN,
    _IDX_DIFF_STD,
    _IDX_DIFF_MAX,
]


def build_feature_matrix(dataset):
    features = []
    velocities = []
    d_values = []
    for sample in dataset.samples:
        x, velocity, d_value = extract_sample_features(sample)
        features.append(x)
        velocities.append(velocity)
        d_values.append(d_value)
    return np.vstack(features), np.array(velocities), np.array(d_values)


def fit_ridge_regression(x_train, y_train, alpha=1e-3):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (x_train - mean) / std
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)

    reg = alpha * np.eye(x_aug.shape[1])
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train)
    return {"mean": mean, "std": std, "weights": weights}


def predict_ridge(model, x):
    x_norm = (x - model["mean"]) / model["std"]
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    return x_aug @ model["weights"]


def get_top_ridge_weights(model, top_k=12):
    weights = model["weights"][1:]
    order = np.argsort(np.abs(weights))[::-1][:top_k]
    rows = []
    for idx in order:
        rows.append(
            {
                "index": int(idx),
                "feature": FEATURE_NAMES[idx],
                "weight": float(weights[idx]),
                "abs_weight": float(abs(weights[idx])),
            }
        )
    return rows


def normalize_features_by_source(x, dataset, feature_indices):
    normalized = x.copy()
    for source, sample_indices in dataset.source_sample_indices.items():
        idx = np.asarray(sample_indices, dtype=np.int64)
        if idx.size == 0:
            continue
        source_x = normalized[np.ix_(idx, feature_indices)]
        mean = source_x.mean(axis=0)
        std = source_x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        normalized[np.ix_(idx, feature_indices)] = (source_x - mean) / std
    return normalized


def build_environment_normalized_matrices(matrices, datasets, feature_indices):
    normalized = {}
    for split_name, values in matrices.items():
        x, y, d_values = values
        normalized[split_name] = (
            normalize_features_by_source(x, datasets[split_name], feature_indices),
            y,
            d_values,
        )
    return normalized


def summarize_by_velocity(x, y, y_pred=None):
    rows = []
    for velocity in sorted(set(y.tolist())):
        mask = y == velocity
        feature_mean = x[mask].mean(axis=0)
        feature_std = x[mask].std(axis=0)
        row = {
            "velocity": velocity,
            "samples": int(mask.sum()),
            "total_events_mean": feature_mean[_IDX_TOTAL_EVENTS],
            "total_events_std": feature_std[_IDX_TOTAL_EVENTS],
            "autocorr_lag1": feature_mean[_IDX_AUTOCORR_LAG1],
            "fano_20us": feature_mean[_IDX_FANO_20US],
            "diff_cv": feature_mean[_IDX_DIFF_CV],
        }
        if y_pred is not None:
            row["pred_mean"] = y_pred[mask].mean()
            row["pred_std"] = y_pred[mask].std()
        rows.append(row)
    return rows


def append_velocity_summary(lines, title, rows, include_pred):
    lines.extend(
        [
            f"### {title}",
            "",
            "| Velocity | Samples | Total Events Mean | Total Events Std | Autocorr Lag1 | Fano 20μs | Diff CV | Pred Mean | Pred Std |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        pred_mean = row.get("pred_mean", float("nan")) if include_pred else float("nan")
        pred_std = row.get("pred_std", float("nan")) if include_pred else float("nan")
        lines.append(
            f"| {row['velocity']:.6f} | {row['samples']} | "
            f"{row['total_events_mean']:.3f} | {row['total_events_std']:.3f} | "
            f"{row['autocorr_lag1']:.6f} | {row['fano_20us']:.4f} | "
            f"{row['diff_cv']:.4f} | {pred_mean:.6f} | {pred_std:.6f} |"
        )
    lines.append("")


def append_metric_summary(lines, split_name, y_true, y_pred):
    mae, rmse, mape = compute_scalar_metrics(y_true, y_pred)
    lines.append(f"- {split_name} MAE: `{mae:.6f}`")
    lines.append(f"- {split_name} RMSE: `{rmse:.6f}`")
    lines.append(f"- {split_name} MAPE: `{mape:.2f}%`")


def append_ridge_weight_summary(lines, ridge):
    lines.extend(
        [
            "## Ridge Feature Weights",
            "",
            "These weights are coefficients after train-set feature standardization inside ridge regression.",
            "",
            "| Rank | Index | Feature | Weight | Abs Weight |",
            "| ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(get_top_ridge_weights(ridge), start=1):
        lines.append(
            f"| {rank} | {row['index']} | `{row['feature']}` | "
            f"{row['weight']:.6f} | {row['abs_weight']:.6f} |"
        )
    lines.append("")


def append_environment_normalized_summary(lines, matrices, predictions, feature_indices):
    feature_list = ", ".join(f"`{FEATURE_NAMES[idx]}`" for idx in feature_indices)
    lines.extend(
        [
            "## Environment-Normalized Ridge Baseline",
            "",
            f"- Environment-normalized features: {feature_list}",
            "- Normalization is done independently inside each source environment before fitting/prediction.",
            "",
        ]
    )
    append_metric_summary(lines, "Train", matrices["train"][1], predictions["train"])
    append_metric_summary(lines, "Val", matrices["val"][1], predictions["val"])
    append_metric_summary(lines, "Eval", matrices["eval"][1], predictions["eval"])
    lines.append("")

    for split_name in ["train", "val", "eval"]:
        x, y, _ = matrices[split_name]
        rows = summarize_by_velocity(x, y, predictions[split_name])
        append_velocity_summary(lines, f"{split_name.title()} Per Velocity After Environment Normalization", rows, include_pred=True)


def write_report(report_path, run_info, matrices, predictions, ridge, env_norm_matrices, env_norm_predictions):
    lines = [
        "# Signal Diagnostics Report",
        "",
        f"- Run timestamp: `{run_info['timestamp']}`",
        f"- total_steps: `{run_info['total_steps']}`",
        f"- dt_us: `{run_info['dt_us']}`",
        f"- mask_path: `{run_info['mask_path']}`",
        f"- ridge_alpha: `{run_info['ridge_alpha']}`",
        "",
        "## Dataset Summary",
        "",
        f"- train_samples: `{len(run_info['train_ds'])}`",
        f"- val_samples: `{len(run_info['val_ds'])}`",
        f"- eval_samples: `{len(run_info['eval_ds'])}`",
        "",
        "## Ridge Baseline",
        "",
    ]

    append_metric_summary(lines, "Train", matrices["train"][1], predictions["train"])
    append_metric_summary(lines, "Val", matrices["val"][1], predictions["val"])
    append_metric_summary(lines, "Eval", matrices["eval"][1], predictions["eval"])
    lines.append("")

    append_ridge_weight_summary(lines, ridge)

    for split_name in ["train", "val", "eval"]:
        x, y, _ = matrices[split_name]
        rows = summarize_by_velocity(x, y, predictions[split_name])
        append_velocity_summary(lines, f"{split_name.title()} Per Velocity", rows, include_pred=True)

    append_environment_normalized_summary(
        lines,
        env_norm_matrices,
        env_norm_predictions,
        ENV_NORMALIZE_FEATURE_INDICES,
    )

    lines.extend(
        [
            "## Feature Names",
            "",
            "| Index | Feature |",
            "| ---: | --- |",
        ]
    )
    for idx, name in enumerate(FEATURE_NAMES):
        lines.append(f"| {idx} | `{name}` |")
    lines.append("")

    lines.extend(
        [
            "## How To Read This",
            "",
            "- If event statistics vary clearly by velocity and the ridge baseline works, the signal exists in preprocessing and the neural model is likely suppressing it.",
            "- If event statistics and ridge predictions are also nearly constant, the selected ROI/time window/preprocessing may not preserve velocity information.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_signal_diagnostics():
    total_steps = 5000
    dt_us = 20
    ridge_alpha = 1e-3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_env_config = {
        "/data/zm/Moshaboli/new_data/no1": 0.018938,
        "/data/zm/Moshaboli/new_data/no4": 0.01973,
        "/data/zm/Moshaboli/new_data/no2": 0.01942,
    }
    val_env_config = {
        "/data/zm/Moshaboli/new_data/no3": 0.01963,
    }
    eval_env_config = {
        "/data/zm/Moshaboli/new_data/no5": 0.01978,
    }
    mask_path = "/data/zm/Moshaboli/new_data/other_data/3.0_mask (2)_hot_pixel_mask.npy"
    report_dir = "/data/zm/Moshaboli/new_data/Markdown"
    report_path = os.path.join(report_dir, f"signal_diagnostics_{timestamp}.md")
    os.makedirs(report_dir, exist_ok=True)

    train_ds = FlexibleBloodFlowDataset(train_env_config, mask_path=mask_path, T=1, seq_len=total_steps, dt_us=dt_us, max_velocity=2.0)
    val_ds = FlexibleBloodFlowDataset(val_env_config, mask_path=mask_path, T=1, seq_len=total_steps, dt_us=dt_us, max_velocity=2.0)
    eval_ds = FlexibleBloodFlowDataset(eval_env_config, mask_path=mask_path, T=1, seq_len=total_steps, dt_us=dt_us, max_velocity=2.0)

    matrices = {
        "train": build_feature_matrix(train_ds),
        "val": build_feature_matrix(val_ds),
        "eval": build_feature_matrix(eval_ds),
    }
    datasets = {
        "train": train_ds,
        "val": val_ds,
        "eval": eval_ds,
    }

    ridge = fit_ridge_regression(matrices["train"][0], matrices["train"][1], alpha=ridge_alpha)
    predictions = {
        split_name: predict_ridge(ridge, values[0])
        for split_name, values in matrices.items()
    }
    env_norm_matrices = build_environment_normalized_matrices(
        matrices,
        datasets,
        ENV_NORMALIZE_FEATURE_INDICES,
    )
    env_norm_ridge = fit_ridge_regression(
        env_norm_matrices["train"][0],
        env_norm_matrices["train"][1],
        alpha=ridge_alpha,
    )
    env_norm_predictions = {
        split_name: predict_ridge(env_norm_ridge, values[0])
        for split_name, values in env_norm_matrices.items()
    }

    write_report(
        report_path,
        {
            "timestamp": timestamp,
            "total_steps": total_steps,
            "dt_us": dt_us,
            "ridge_alpha": ridge_alpha,
            "mask_path": mask_path,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "eval_ds": eval_ds,
        },
        matrices,
        predictions,
        ridge,
        env_norm_matrices,
        env_norm_predictions,
    )
    print(f"Saved signal diagnostics report to: {report_path}")


if __name__ == "__main__":
    run_signal_diagnostics()
