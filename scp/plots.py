import matplotlib.pyplot as plt
import numpy as np

import scp.metrics as metrics

"""
    Common plots used in multiple notebooks
"""

#############################################
# scvi-tools plots
#############################################


def plot_loss(history, epoch_start=0, validation_smooth_window=None, pad=3):
    fig, axes = plt.subplots(figsize=(16, 4), ncols=3, nrows=1)
    fig.tight_layout(pad=pad)

    def _plot(key, color, ax):
        if key not in history:
            return

        series = history[key]
        offset = series.index >= epoch_start
        ax.plot(series.iloc[offset], color=color)

    def _smooth(history, key, window=10):
        if key not in history:
            return

        new_key = f"{key}_smooth"
        history[new_key] = history[key].rolling(window).median()
        history[new_key].iloc[:window] = history[key].iloc[:window]

    if validation_smooth_window is not None:
        _smooth(history, "validation_loss", window=validation_smooth_window)
        _smooth(
            history, "reconstruction_loss_validation", window=validation_smooth_window
        )
        _smooth(history, "kl_local_validation", window=validation_smooth_window)

    smooth_postfix = "_smooth" if validation_smooth_window is not None else ""

    ax = axes[0]
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Total loss")
    ax.grid(True)
    ax.set_axisbelow(True)
    _plot("train_loss_epoch", "blue", ax)
    _plot("validation_loss" + smooth_postfix, "orange", ax)

    ax = axes[1]
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.set_title("Reconstruction loss")
    ax.grid(True)
    ax.set_axisbelow(True)
    _plot("reconstruction_loss_train", "blue", ax)
    _plot("reconstruction_loss_validation" + smooth_postfix, "orange", ax)

    ax = axes[2]
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL-divergence")
    ax.set_title("KL-divergence")
    ax.grid(True)
    ax.set_axisbelow(True)
    _plot("kl_local_train", "blue", ax)
    _plot("kl_local_validation" + smooth_postfix, "orange", ax)

    ax.legend(["Train", "Validation"], loc="upper right", markerscale=2)


#############################################
# generic results plots
#############################################


def scatter_protein_detection_proportion_and_intensity(x, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x_obs_protein = np.nanmean(x, axis=0)
    p_protein = 1 - np.mean(np.isnan(x), axis=0)

    ax.scatter(x_obs_protein, p_protein, color="blue", alpha=1, s=6)
    ax.set_title(title)
    ax.set_xlabel("protein log-intensity")
    ax.set_ylabel("detection proportion")
    ax.grid(True)
    ax.set_axisbelow(True)


def scatter_sample_mean_and_variance(x, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sample_mean = np.nanmean(x, axis=1)
    sample_var = np.nanvar(x, axis=1)

    ax.scatter(sample_mean, sample_var, color="blue", s=2)
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Sample variance")
    ax.set_title(title)
    ax.grid(True)
    ax.set_axisbelow(True)


def scatter_protein_mean_and_cv(x, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    mean_protein = np.nanmean(x, axis=0)
    mean_std = np.nanstd(x, axis=0)
    cv_protein = mean_std / mean_protein

    ax.scatter(mean_protein, cv_protein, color="blue", s=2)
    ax.set_xlabel("Protein mean")
    ax.set_ylabel("Protein CV")
    ax.set_title(title)
    ax.grid(True)
    ax.set_axisbelow(True)


def plot_protein_detection_proportion_panel(x, p_est, x_est=None, color="blue", title="PROTVI"):
    x_protein = np.nanmean(x, axis=0)
    p_protein = 1 - np.mean(np.isnan(x), axis=0)
    p_est_protein = p_est.mean(axis=0)

    x_est_protein = np.nanmean(x_est, axis=0) if x_est is not None else None

    fig, axes = plt.subplots(figsize=(20, 5), ncols=4)
    fig.suptitle(title, fontsize=16, y=1.03)
    fig.tight_layout(w_pad=3)

    _scatter_compare_protein_detection_proportion_and_intensity(
        x_protein, p_protein, p_est_protein, x_est_protein=x_est_protein, color=color, ax=axes[0]
    )

    _scatter_compare_protein_detection_proportion(
        p_protein, p_est_protein, color=color, ax=axes[1]
    )
    _hist_compare_protein_detection_proportion_difference(
        p_protein, p_est_protein, color=color, ax=axes[2]
    )
    _scatter_compare_protein_detection_proportion_difference(
        x_protein, p_protein, p_est_protein, color=color, ax=axes[3]
    )


def _scatter_compare_protein_detection_proportion_and_intensity(
    x_protein, p_protein, p_est_protein, x_est_protein=None, color="blue", ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        x_protein,
        p_protein,
        color="black",
        s=4,
        alpha=0.4,
        label="Observed",
    )
    ax.scatter(
        x_est_protein if x_est_protein is not None else x_protein,
        p_est_protein,
        color=color,
        s=4,
        alpha=0.4,
        label="Predicted",
    )
    ax.set_xlabel("Avg. observed log-intensity")
    ax.set_ylabel("Detection proportion")
    ax.set_title("Protein log-intensity vs. detection proportion")
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(markerscale=2)


def _scatter_compare_protein_detection_proportion(
    p_protein, p_est_protein, color="blue", ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    mse = metrics.mse(p_protein, p_est_protein)
    ax.text(
        0.03,
        0.94,
        f"MSE: {mse:.3f}",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        transform=ax.transAxes,
    )

    ax.scatter(
        p_protein,
        p_est_protein,
        color=color,
        edgecolor="black",
        linewidth=0,
        s=6,
        alpha=0.5,
    )
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.2, linestyle="--")
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.set_xlabel("Observed detection proportion")
    ax.set_ylabel("Predicted detection proportion")
    ax.set_title("Observed vs. predicted protein detection proportion")
    ax.grid(True)
    ax.set_axisbelow(True)


def _hist_compare_protein_detection_proportion_difference(
    p_protein, p_est_protein, color="blue", ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    diff = p_est_protein - p_protein
    ax.hist(diff, bins=60, color=color, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Number of proteins")
    ax.set_xlabel("Predicted - observed detection proportion")
    ax.set_title("Difference in detection proportion between\n observed and predicted")
    ax.grid(True)
    ax.set_axisbelow(True)


def _scatter_compare_protein_detection_proportion_difference(
    x_protein, p_protein, p_est_protein, color="blue", ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    diff = p_est_protein - p_protein
    ax.scatter(
        x_protein, diff, color=color, edgecolor="black", linewidth=0, s=6, alpha=0.5
    )
    ax.set_xlabel("Avg. observed log-intensity")
    ax.set_ylabel("Predicted - observed detection proportion")
    ax.set_title("Difference in detection proportion \nvs. avg. protein intensity")
    ax.grid(True)
    ax.set_axisbelow(True)


def plot_protein_intensity_panel(x, x_est, title="PROTVI"):
    fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
    fig.tight_layout(pad=4)
    fig.suptitle(title, fontsize=16, y=1.05)

    x_est_obs = x_est.copy()
    x_est_obs[np.isnan(x)] = np.nan

    x_est_miss = x_est.copy()
    x_est_miss[~np.isnan(x)] = np.nan

    x_obs_protein = np.nanmean(x, axis=0)
    x_est_obs_protein = np.nanmean(x_est_obs, axis=0)

    mse = metrics.mse(x_obs_protein, x_est_obs_protein)

    ax = axes[0]
    ax.plot(
        [0, x_obs_protein.max()],
        [0, x_est_obs_protein.max()],
        color="black",
        linewidth=1.2,
        linestyle="--",
    )
    ax.scatter(
        x_obs_protein,
        x_est_obs_protein,
        color="red",
        edgecolor="black",
        linewidth=0,
        s=6,
        alpha=0.5,
    )
    ax.scatter(
        np.mean(x_obs_protein),
        np.mean(x_est_obs_protein),
        color="cyan",
        edgecolor="black",
        linewidth=1,
        s=40,
        alpha=1,
    )
    ax.text(
        0.03,
        0.94,
        f"MSE: {mse:.3f}",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        transform=ax.transAxes,
    )
    ax.set_xlabel("Observed protein log-intensity")
    ax.set_ylabel("Predicted protein log-intensity")
    ax.set_title("Predicted intensity vs. observed intensity \n- for each protein")
    ax.grid(True)
    ax.set_axisbelow(True)

    #############################################

    protein_with_missing_intensities_mask = np.isnan(x).any(axis=0)
    x_est_obs_protein_shared = np.nanmean(
        x_est_obs[:, protein_with_missing_intensities_mask], axis=0
    )
    x_est_miss_protein_shared = np.nanmean(
        x_est_miss[:, protein_with_missing_intensities_mask], axis=0
    )

    ax = axes[1]
    min_v = min(x_est_obs_protein_shared.min(), x_est_miss_protein_shared.min())
    max_v = max(x_est_obs_protein_shared.max(), x_est_miss_protein_shared.max())
    ax.plot(
        [min_v, max_v], [min_v, max_v], color="black", linewidth=1.2, linestyle="--"
    )
    ax.scatter(
        x_est_miss_protein_shared,
        x_est_obs_protein_shared,
        color="red",
        edgecolor="black",
        linewidth=0,
        s=6,
        alpha=0.5,
    )
    ax.scatter(
        np.mean(x_est_miss_protein_shared),
        np.mean(x_est_obs_protein_shared),
        color="cyan",
        edgecolor="black",
        linewidth=1,
        s=40,
        alpha=1,
    )
    ax.set_xlabel("Predicted missing protein log-intensity")
    ax.set_ylabel("Predicted observed protein log-intensity")
    ax.set_title(
        "Predicted observed intensity vs. predicted missing intensity \n- for each protein"
    )
    ax.grid(True)
    ax.set_axisbelow(True)

    ax = axes[2]
    diff = x_est_obs_protein_shared - x_est_miss_protein_shared
    ax.hist(diff, bins=60, color="red", edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Number of proteins")
    ax.set_xlabel(
        "Avg. pred. observed log-intensity - avg. pred. missing log-intensity"
    )
    ax.set_title(
        "Difference in predicted observed intensity vs. \n predicted missing intensity - for each protein"
    )
    ax.grid(True)
    ax.set_axisbelow(True)


def scatter_compare_protein_missing_intensity(
    x_protein, x_est_protein, color="red", ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        x_protein,
        x_est_protein,
        color=color,
        edgecolor="black",
        linewidth=0,
        s=6,
        alpha=0.5,
    )
    v_min = min(x_protein.min(), x_est_protein.min())
    v_max = max(x_protein.max(), x_est_protein.max())
    ax.plot(
        [v_min, v_max], [v_min, v_max], color="black", linewidth=1.2, linestyle="--"
    )

    mse = metrics.mse(x_protein, x_est_protein)
    ax.text(
        0.03,
        0.94,
        f"MSE: {mse:.3f}",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        transform=ax.transAxes,
    )

    pearson = metrics.pearson(x_protein, x_est_protein)
    ax.text(
        0.03,
        0.83,
        f"Pearson corr.\n{pearson:.3f}",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        transform=ax.transAxes,
    )

    ax.set_xlabel("Observed missing protein log-intensity")
    ax.set_ylabel("Predicted missing protein log-intensity")
    ax.set_title(
        "Predicted missing intensity vs. observed missing intensity \n- for each protein"
    )
    ax.grid(True)
    ax.set_axisbelow(True)


def plot_model_intensity_comparison(
    x,
    x_est_obs_protein1,
    x_est_miss_protein1,
    x_est_obs_protein2,
    x_est_miss_protein2,
    model1_name,
    model2_name,
):
    proteins_with_missing_intensities_mask = np.isnan(x).any(axis=0)

    x_est_miss_protein1 = x_est_miss_protein1[proteins_with_missing_intensities_mask]
    x_est_obs_protein1 = x_est_obs_protein1[proteins_with_missing_intensities_mask]

    x_est_obs_protein2 = x_est_obs_protein2[proteins_with_missing_intensities_mask]
    x_est_miss_protein2 = x_est_miss_protein2[proteins_with_missing_intensities_mask]

    from sklearn.linear_model import LinearRegression

    lm = LinearRegression()

    fig, axes = plt.subplots(figsize=(18, 6), ncols=3)
    fig.tight_layout(pad=3)
    fig.suptitle(f"{model1_name} vs {model2_name}", fontsize=16, y=1)

    ax = axes[0]
    ax.scatter(
        x_est_miss_protein1,
        x_est_obs_protein1,
        color="blue",
        edgecolor="black",
        linewidth=0,
        s=4,
        alpha=0.3,
    )
    ax.scatter(
        x_est_miss_protein2,
        x_est_obs_protein2,
        color="red",
        edgecolor="purple",
        linewidth=0,
        s=4,
        alpha=0.3,
    )

    v1 = min(
        x_est_obs_protein1.min(),
        x_est_miss_protein1.min(),
        x_est_obs_protein2.min(),
        x_est_miss_protein2.min(),
    )
    v2 = max(
        x_est_obs_protein1.max(),
        x_est_miss_protein1.max(),
        x_est_obs_protein2.max(),
        x_est_miss_protein2.max(),
    )

    ax.plot(
        [v1, v2],
        [v1, v2],
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
    )

    ## model 1
    lm.fit(x_est_miss_protein1.reshape(-1, 1), x_est_obs_protein1.reshape(-1, 1))
    ax.plot(
        [v1, v2],
        [
            lm.intercept_[0] + lm.coef_[0] * v1,
            lm.intercept_[0] + lm.coef_[0] * v2,
        ],
        color="black",
        linewidth=4,
        linestyle="-",
        solid_capstyle="round",
    )
    ax.plot(
        [v1, v2],
        [
            lm.intercept_[0] + lm.coef_[0] * v1,
            lm.intercept_[0] + lm.coef_[0] * v2,
        ],
        color="blue",
        linewidth=2,
        linestyle="-",
        solid_capstyle="round",
        label=model1_name,
    )

    ## model 2
    lm.fit(x_est_miss_protein2.reshape(-1, 1), x_est_obs_protein2.reshape(-1, 1))
    ax.plot(
        [v1, v2],
        [
            lm.intercept_[0] + lm.coef_[0] * v1,
            lm.intercept_[0] + lm.coef_[0] * v2,
        ],
        color="black",
        linewidth=3,
        linestyle="-",
        solid_capstyle="round",
    )
    ax.plot(
        [v1, v2],
        [
            lm.intercept_[0] + lm.coef_[0] * v1,
            lm.intercept_[0] + lm.coef_[0] * v2,
        ],
        color="red",
        linewidth=2,
        linestyle="-",
        solid_capstyle="round",
        label=model2_name,
    )

    ax.set_xlabel("Predicted missing protein log-intensity")
    ax.set_ylabel("Predicted observed protein log-intensity")
    ax.set_title(
        "Predicted missing intensity vs. predicted observed intensity - for each protein"
    )
    ax.legend(markerscale=2)
    ax.grid(True)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.scatter(
        x_est_miss_protein1,
        x_est_miss_protein2,
        color="blue",
        edgecolor="purple",
        linewidth=0,
        s=4,
        alpha=0.3,
    )
    v1 = min(x_est_miss_protein2.min(), x_est_miss_protein1.min())
    v2 = max(x_est_miss_protein2.max(), x_est_miss_protein1.max())
    ax.plot(
        [v1, v2],
        [v1, v2],
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
    )
    ax.set_xlabel(f"Missing protein log-intensity ({model1_name})")
    ax.set_ylabel(f"Missing protein log-intensity ({model2_name})")
    ax.set_title("Missing intensity - for each protein")
    ax.grid(True)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.scatter(
        x_est_obs_protein1,
        x_est_obs_protein2,
        color="blue",
        edgecolor="purple",
        linewidth=0,
        s=4,
        alpha=0.3,
    )
    v1 = min(x_est_obs_protein2.min(), x_est_obs_protein1.min())
    v2 = max(x_est_obs_protein2.max(), x_est_obs_protein1.max())
    ax.plot(
        [v1, v2],
        [v1, v2],
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
    )
    ax.set_xlabel(f"Predicted observed protein log-intensity ({model1_name})")
    ax.set_ylabel(f"Predicted observed protein log-intensity ({model2_name})")
    ax.set_title("Observed intensity - for each protein")
    ax.grid(True)
    ax.set_axisbelow(True)


#############################################
# protDP plots
#############################################


# @TODO: consider creating a ProtDP wrapper class for protdp_result
def plot_protein_detection_proportion_panel_protDP(
    x, protdp_result, color="blue", title="protDP"
):
    beta = protdp_result["beta"]
    beta_start = protdp_result["betaStart"]

    x_protein = np.nanmean(x, axis=0)
    p_protein = 1 - np.mean(np.isnan(x), axis=0)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    fig, axes = plt.subplots(figsize=(18, 5), ncols=4)
    fig.tight_layout(pad=4)
    fig.suptitle(title, fontsize=16, y=1.05)

    ax = axes[0]
    min_intensity = np.nanmin(x_protein)
    max_intensity = np.nanmax(x_protein)
    x_step = np.linspace(min_intensity, max_intensity, 100)
    y1 = sigmoid(beta_start[0] + beta_start[1] * x_step)
    y2 = sigmoid(beta[0] + beta[1] * x_step)
    ax.plot(x_step, y1, color="black", linewidth=1.2, linestyle="--", label="Start")
    ax.plot(x_step, y2, color=color, linewidth=1.2, linestyle="--", label="Final")
    ax.scatter(x_protein, p_protein, color="black", linewidth=0, s=2, alpha=0.5)
    ax.set_xlabel("Avg. protein log-intensity")
    ax.set_ylabel("Detection proportion")
    ax.set_title("Detection proportion vs. avg. protein intensity")
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True)
    ax.legend()
    ax.set_axisbelow(True)

    p_est = sigmoid(beta[0] + beta[1] * x_protein)

    _scatter_compare_protein_detection_proportion(
        p_protein, p_est, color=color, ax=axes[1]
    )
    _hist_compare_protein_detection_proportion_difference(
        p_protein, p_est, color=color, ax=axes[2]
    )
    _scatter_compare_protein_detection_proportion_difference(
        x_protein, p_protein, p_est, color=color, ax=axes[3]
    )
