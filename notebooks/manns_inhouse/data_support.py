import pandas as pd
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import scp.utils as utils
import scp.plots as pl
import scp.metrics as metrics

"""
    Functionality specific for the Mann's lab data.
"""


# data loading
import os 

def load_main_data(dir: str):
    annotations_path = os.path.join(dir, "annotations_main_lt_v21_Sc11_AIMsplit3.tsv")
    data_path = os.path.join(
        dir,
        "DA-F08.4_-SEC-pass_v06Sc_ion_LibPGQVal1perc_precdLFQdefFull_prot_preprSc03.tsv",
    )
    stats_path = os.path.join(
        dir, "DA-F08.4_PL01-POL-LT-QC2_DA-SER1.19_C20Lib02_.stats.tsv"
    )

    data = pd.read_csv(data_path, sep="\t", index_col="protein")
    obs = pd.read_csv(annotations_path, sep="\t")
    stats = pd.read_csv(stats_path, sep="\t")

    data.drop("Unnamed: 0", axis=1, inplace=True)
    obs.drop("Unnamed: 0", axis=1, inplace=True)

    # extract vars
    var_cols = [c for c in data.columns if "JaBa" not in c]
    vars = data[var_cols]
    data.drop(var_cols, axis=1, inplace=True)

    data = data.T

    # extract the "run number" from the run name - which is unique shared between the files
    data["filename"] = [run.split("/")[-1].split(".")[0] for run in data.index]
    data.set_index("filename", drop=True, inplace=True, verify_integrity=True)

    obs["filename"] = [run.split("/")[-1].split(".")[0] for run in obs["Run"]]
    obs.set_index("filename", drop=True, inplace=True, verify_integrity=True)

    stats["filename"] = [run.split("/")[-1].split(".")[0] for run in stats["File.Name"]]
    stats.set_index("filename", drop=True, inplace=True, verify_integrity=True)

    drop = [
        "Comment",
        "CVsampleInfo_I",
        "CVsampleInfo_II",
        "CVsampleInfo_III",
        "QC_Experiment",
        "QC_Sample_ID",
        "QC_Patient_ID",
        "QC_Condition",
        "QC_Condition_numeric",
        "PatientID_LT",
        "N_puncture_LT",
        "Diff_to_first_puncture_LT",
        "ID_MAIN_LT",
    ]
    obs = obs.drop(drop, axis=1)

    # ignore LT, QC, PO plates and plates with multiple runs
    obs = obs[obs["Sample_plate"].str.match(r"^plate\d+$")]

    # remove more quality control samples
    obs = obs[["Pool" not in obs for obs in obs["MSgroup"]]]

    for o in [
        "Leukocyte_count",
        "Albumin_CSF",
        "QAlb",
        "IgG_CSF",
        "QIgG",
        "Total_protein",
    ]:
        obs[o] = [
            np.nan if a in ["n. best.", "n. best. ", "na", "not measured"] else float(a)
            for a in obs[o]
        ]
    obs["Erythrocytes_in_CSF"] = [
        np.nan if a in ["n. best.", "n. best. ", "na", "not measured"] else a
        for a in obs["Erythrocytes_in_CSF"]
    ]

    obs["Total_protein"][obs["Total_protein"] == 0] = np.nan
    obs["Diagnosis_group_subtype"][obs["Diagnosis_group_subtype"] == "unknown"] = np.nan

    obs["Evosept"] = [a.split("_")[4][1] for a in obs["Run"]]
    obs["Column"] = [a.split("_")[4][3] for a in obs["Run"]]
    obs["Emitter"] = [a.split("_")[4][5] for a in obs["Run"]]
    obs["Capillary"] = [a.split("_")[4][7] for a in obs["Run"]]
    obs["Maintenance"] = [a.split("_")[4][9:11] for a in obs["Run"]]

    obs["Age"] = obs["Age"].astype("float")
    obs["log Qalb"] = np.log(obs["QAlb"])

    obs.rename(
        {
            "Leukocyte_count": "Leukocyte count",
            "Total_protein": "Total protein",
            "IgG_CSF": "IgG CSF",
            "QAlb": "Qalb",
            "Albumin_CSF": "Albumin CSF",
            "Erythrocytes_in_CSF": "Erythrocytes",
            "Sample_plate": "Plate",
            "Sample_preparation_batch": "Preparation day",
        },
        axis=1,
        inplace=True,
    )

    ## create adata from data, vars, obs
    obs = pd.merge(obs, stats, how="inner", left_index=True, right_index=True)
    obs = pd.merge(
        pd.DataFrame(index=data.index),
        obs,
        how="inner",
        left_index=True,
        right_index=True,
    )

    data = data.loc[obs.index]

    adata = sc.AnnData(data, var=vars, obs=obs)
    adata.obs = adata.obs.set_index("ID", drop=True, verify_integrity=True)

    adata.strings_to_categoricals()

    adata = adata[~adata.obs["Qalb"].isna()].copy()
    mask = np.array([e not in ["++", "+++", "bloody"] for e in adata.obs["Erythrocytes"]])
    adata = adata[mask]
    adata = adata[~adata.obs["Erythrocytes"].isna()]

    adata.obs["diagnosis_group_autoimmune"] = adata.obs["Diagnosis_group_autoimmune_split"].replace("Autoimmune_notMSrelated", "Autoimmune").replace("Autoimmune_MSrelated", "Autoimmune")
    adata.obs["diagnosis_group_autoimmune"] = adata.obs["diagnosis_group_autoimmune"].astype("category")

    adata.obs["Preparation day"] = [f"day{int(x[-1])}" for x in adata.obs["Preparation day"]]

    return adata


def load_pilot_data(dir: str):
    ANNOTATIONS_PATH = os.path.join(
        dir, "2023_sample annotation_PILOTcohorts combined_v21_Sc06.tsv"
    )
    DATA_PATH = os.path.join(
        dir,
        "DAP-F03.4_-SEC-pass_v06Sc_ion_LibPGQVal1perc_precdLFQdefFull_protein_intensities.tsv",
    )
    STATS_PATH = os.path.join(
        dir, "DAP-F03.4_CH1+2+QC_DA-SER2.01_LibC20Lib02_.stats.tsv"
    )

    data = pd.read_csv(DATA_PATH, sep="\t", index_col="protein")
    obs = pd.read_csv(ANNOTATIONS_PATH, sep="\t")
    stats = pd.read_csv(STATS_PATH, sep="\t")

    obs.drop("Unnamed: 0", axis=1, inplace=True)

    # extract vars
    var_cols = [c for c in data.columns if "JaBa" not in c]
    vars = data[var_cols]
    data.drop(var_cols, axis=1, inplace=True)

    data = data.T

    data["filename"] = [run.split("/")[-1].split(".")[0] for run in data.index]
    data.set_index("filename", drop=True, inplace=True, verify_integrity=True)

    obs["filename"] = [run.split("/")[-1].split(".")[0] for run in obs["Run"]]
    obs.set_index("filename", drop=True, inplace=True, verify_integrity=True)

    stats["filename"] = [run.split("/")[-1].split(".")[0] for run in stats["File.Name"]]
    stats.set_index("filename", drop=True, inplace=True, verify_integrity=True)

    # log normalize intensities and set 0's to nan to be consistent between pilot and main dataset.
    data[data == 0] = np.nan
    data = np.log(data)

    # filter out plate 5 since in the PILOT some samples were measured multiple times. Recommended by Jakob at Manns lab.
    obs = obs[obs["Sample_plate"] != "plate5"]

    # remove more quality control samples
    obs = obs[~obs["MSgroup"].isna()]
    obs = obs[["Pool" not in obs for obs in obs["MSgroup"]]]

    # for some reason, "QCpool_PILOT1" spilled over to "Age".
    obs = obs[["pool" not in obs for obs in obs["Age"]]]

    # remove more quality control samples
    obs = obs[["Pool" not in obs for obs in obs["MSgroup"]]]

    for o in [
        "Leukocyte_count",
        "Albumin_CSF",
        "QAlb",
        "IgG_CSF",
        "QIgG",
        "Total_protein",
    ]:
        obs[o] = [
            np.nan if a in ["n. best.", "n. best. ", "na", "not measured"] else float(a)
            for a in obs[o]
        ]
    obs["Erythrocytes_in_CSF"] = [
        np.nan if a in ["n. best.", "n. best. ", "na", "not measured"] else a
        for a in obs["Erythrocytes_in_CSF"]
    ]

    obs["Total_protein"][obs["Total_protein"] == 0] = np.nan
    obs["Diagnosis_group_subtype"][obs["Diagnosis_group_subtype"] == "unknown"] = np.nan

    obs["Age"] = obs["Age"].astype("float")
    obs["log Qalb"] = np.log(obs["QAlb"])

    obs.rename(
        {
            "Leukocyte_count": "Leukocyte count",
            "Total_protein": "Total protein",
            "IgG_CSF": "IgG CSF",
            "QAlb": "Qalb",
            "Albumin_CSF": "Albumin CSF",
            "Erythrocytes_in_CSF": "Erythrocytes",
            "Sample_plate": "Plate",
            "Sample_preparation_batch": "Preparation day",
        },
        axis=1,
        inplace=True,
    )

    # remove duplicate samples based on ID_main
    id_main_counts = obs["ID_main"].value_counts()
    id_main_unique_idx = id_main_counts[id_main_counts == 1].index
    obs = obs[obs["ID_main"].isin(id_main_unique_idx)]

    ## create adata from data, vars, obs
    obs = pd.merge(obs, stats, how="inner", left_index=True, right_index=True)
    obs = pd.merge(
        pd.DataFrame(index=data.index),
        obs,
        how="inner",
        left_index=True,
        right_index=True,
    )

    data = data.loc[obs.index]

    adata = sc.AnnData(data, var=vars, obs=obs)
    adata.obs = adata.obs.set_index("ID_main", drop=True, verify_integrity=True)

    adata.strings_to_categoricals()

    adata = adata[~adata.obs["Qalb"].isna()].copy()
    mask = np.array([e not in ["++", "+++", "bloody"] for e in adata.obs["Erythrocytes"]])
    adata = adata[mask]
    adata = adata[~adata.obs["Erythrocytes"].isna()]

    return adata


def filter_by_detection_proportion_by_patient(adata, min_protein_completeness=0.2):
    adata.var["filter"] = 1
    for group in np.unique(adata.obs["Diagnosis_group"]):
        sub = adata[adata.obs["Diagnosis_group"] == group]
        completeness = (sub.X == sub.X).mean(axis=0)
        adata.var["filter"] = adata.var["filter"] * (
            completeness < min_protein_completeness
        ).astype(int)

    adata.var["filter"] = adata.var["filter"].astype(bool)
    adata._inplace_subset_var(~adata.var["filter"])


def preprocess(adata, filter_cells=0, min_protein_completeness=0.2, verbose=True):
    if verbose:
        print(f"preprocess input: {adata.shape}")

    sc.pp.filter_genes(adata, min_cells=1)
    if verbose:
        print(f"sc.pp.filter_genes: {adata.shape}")

    sc.pp.filter_cells(adata, min_genes=filter_cells)
    if verbose:
        print(f"sc.pp.filter_cells: {adata.shape}")

    filter_by_detection_proportion_by_patient(
        adata, min_protein_completeness=min_protein_completeness
    )
    if verbose:
        print(f"filter: {adata.shape}")

    return adata


def correct_batch(adata):
    """Corrects batch for anndata.X - keeping the original reference to X"""
    mean_protein = np.nanmean(adata.X, axis=0)

    plates = np.unique(adata.obs["Plate"][~adata.obs["Plate"].isna()])

    stds_plates = []
    for plate in plates:
        plate_adata = adata[adata.obs["Plate"] == plate]
        std_protein_per_plate = np.nanstd(plate_adata.X, axis=0)
        stds_plates.append(std_protein_per_plate)

    std = np.nanmean(stds_plates)

    for plate in plates:
        mask = (adata.obs["Plate"] == plate).values
        x = adata.X[mask, :]

        mean_protein_per_plate = np.nanmean(x, axis=0)
        std_protein_per_plate = np.nanstd(x, axis=0)

        # If only 1 cell exists in the plate, std_protein_per_plate will be 0. So we correct this.
        std_protein_per_plate[std_protein_per_plate == 0] = std

        adata.X[mask, :] = (
            ((x - mean_protein_per_plate) / std_protein_per_plate) * std
        ) + mean_protein

    # result stored in adata.X


def integrate_dataset(x_pilot, x_main):
    """regress difference between pilot and main out using overlapping intensities"""
    overlap_mask = ~np.isnan(x_pilot) & ~np.isnan(x_main)
    row_idx, col_idx = np.where(overlap_mask)

    # fit linear curve
    x_pilot_fit = x_pilot[row_idx, col_idx]
    x_main_fit = x_main[row_idx, col_idx]

    import scipy.stats as stats

    slope, intercept, _, _, _ = stats.linregress(x_pilot_fit, x_main_fit)
    x_pilot_l = slope * x_pilot.copy() + intercept

    return x_pilot_l, x_main.copy()


def load_data(
    main_dir: str,
    pilot_dir: str,
    do_batch_correction: bool = False,
    integrate: bool = False,
    verbose: bool = True,
):
    main_adata = load_main_data(main_dir)
    main_adata = preprocess(main_adata, verbose=verbose)

    pilot_adata = load_pilot_data(pilot_dir)
    pilot_adata = preprocess(pilot_adata, verbose=verbose)

    if do_batch_correction:
        correct_batch(main_adata)
        correct_batch(pilot_adata)

    pilot_adata = utils.reshape_anndata_like(adata=pilot_adata, adata_like=main_adata)

    if integrate:
        pilot_adata.X, main_adata.X = integrate_dataset(pilot_adata.X, main_adata.X)

    adata = main_adata
    adata.layers["main"] = main_adata.X.copy()
    adata.layers["pilot"] = pilot_adata.X.copy()

    if verbose:
        x_combined = utils.fill_if_nan(main_adata.X, pilot_adata.X)

        print()
        print(f"main intensity coverage:     {metrics.get_coverage(main_adata.X):.2%}")
        print(f"pilot intensity coverage:    {metrics.get_coverage(pilot_adata.X):.2%}")
        print(f"combined intensity coverage: {metrics.get_coverage(x_combined):.2%}")

    return adata


def compute_common_metrics(x_main, x_pilot, x_est):
    # remove the entries in pilot which has an entry in main
    x_pilot = x_pilot.copy()
    x_pilot[~np.isnan(x_main)] = np.nan

    m_pilot = ~np.isnan(x_pilot)
    m_main = ~np.isnan(x_main)

    ## entry-wise intensities ##

    # mse
    main_model_mse = metrics.mse(x_main[m_main], x_est[m_main])
    pilot_model_mse = metrics.mse(x_pilot[m_pilot], x_est[m_pilot])

    # pearson
    main_model_pearson = metrics.pearson(x_main[m_main], x_est[m_main])
    pilot_model_pearson = metrics.pearson(x_pilot[m_pilot], x_est[m_pilot])

    ## protein-wise intensities ##
    x_est_main_obs = x_est.copy()
    x_est_main_obs[~m_main] = np.nan

    x_est_pilot_obs = x_est.copy()
    x_est_pilot_obs[~m_pilot] = np.nan

    # keep proteins with at least 1 intensity measurement
    idx_proteins_main = get_protein_idx(x_main, min_protein_replicas=1)
    idx_proteins_pilot = get_protein_idx(x_pilot, min_protein_replicas=1)

    x_main_protein = np.nanmean(x_main[:, idx_proteins_main], axis=0)
    x_pilot_protein = np.nanmean(x_pilot[:, idx_proteins_pilot], axis=0)

    x_est_main_obs_protein = np.nanmean(x_est_main_obs[:, idx_proteins_main], axis=0)
    x_est_pilot_obs_protein = np.nanmean(x_est_pilot_obs[:, idx_proteins_pilot], axis=0)

    # mse
    main_model_protein_mse = metrics.mse(x_main_protein, x_est_main_obs_protein)
    pilot_model_protein_mse = metrics.mse(x_pilot_protein, x_est_pilot_obs_protein)

    # pearson
    main_model_protein_pearson = metrics.pearson(x_main_protein, x_est_main_obs_protein)
    pilot_model_protein_pearson = metrics.pearson(x_pilot_protein, x_est_pilot_obs_protein)

    ## protein-wise difference intensities ##

    # compare difference between estimated protein intensities and pilot in the categories obs and miss - based on main.
    

    result = {
        # entry-wise
        "main_model_mse": main_model_mse,
        "pilot_model_mse": pilot_model_mse,
        "main_model_pearson": main_model_pearson,
        "pilot_model_pearson": pilot_model_pearson,
        # protein-wise
        "main_model_protein_mse": main_model_protein_mse,
        "pilot_model_protein_mse": pilot_model_protein_mse,
        "main_model_protein_pearson": main_model_protein_pearson,
        "pilot_model_protein_pearson": pilot_model_protein_pearson,
    }

    return result


def compute_common_metrics_protDP(x_main, x_pilot, x_est_obs, x_est_miss):
    # also remove the entries in pilot which has an entry in main
    x_pilot = x_pilot.copy()
    x_pilot[~np.isnan(x_main)] = np.nan

    # keep proteins with at least 1 intensity measurement
    idx_proteins_main = get_protein_idx(x_main, min_protein_replicas=1)
    idx_proteins_pilot = get_protein_idx(x_pilot, min_protein_replicas=1)

    x_main_protein = np.nanmean(x_main[:, idx_proteins_main], axis=0)
    x_pilot_protein = np.nanmean(x_pilot[:, idx_proteins_pilot], axis=0)

    x_est_main_protDP = x_est_obs[idx_proteins_main]
    x_est_pilot_protDP = x_est_miss[idx_proteins_pilot]

    # mse
    main_model_protein_mse = np.nanmean((x_main_protein - x_est_main_protDP) ** 2)
    pilot_model_protein_mse = np.nanmean((x_pilot_protein - x_est_pilot_protDP) ** 2)

    # pearson
    main_model_protein_pearson = metrics.pearson(x_main_protein, x_est_main_protDP)
    pilot_model_protein_pearson = metrics.pearson(x_pilot_protein, x_est_pilot_protDP)

    result = {
        "main_model_protein_mse": main_model_protein_mse,
        "pilot_model_protein_mse": pilot_model_protein_mse,
        "main_model_protein_pearson": main_model_protein_pearson,
        "pilot_model_protein_pearson": pilot_model_protein_pearson,
    }

    return result


RESULT_DIR = "../../results/manns_lab_data/"


# saving and loading
def save_dict_to_results(dict, filename, results_dir=RESULT_DIR):
    path = os.path.join(results_dir, filename)
    utils.save_dict(dict, path)


def load_dict_from_results(filename, results_dir=RESULT_DIR):
    path = os.path.join(results_dir, filename)
    return utils.load_dict(path)


# plotting
def get_protein_overlap_idx(x1, x2, n_min_protein_overlap):
    overlap_mask = ~np.isnan(x1) & ~np.isnan(x2)
    n_proteins = overlap_mask.sum(axis=0)
    idx_proteins = np.where(n_proteins >= n_min_protein_overlap)[0]
    return idx_proteins


def get_protein_idx(x, min_protein_replicas):
    n_proteins = (~np.isnan(x)).sum(axis=0)
    idx_proteins = np.where(n_proteins >= min_protein_replicas)[0]
    return idx_proteins


def scatter_main_pilot_model(
    x_main,
    x_pilot,
    x_est,
    model_name="PROTVI",
    metric_types=["pearson", "spearman", "mse"],
    n_min_protein_overlap=2,
):
    idx_proteins_main_pilot = get_protein_overlap_idx(
        x_main, x_pilot, n_min_protein_overlap=n_min_protein_overlap
    )
    idx_proteins_model_pilot = get_protein_overlap_idx(
        x_est, x_pilot, n_min_protein_overlap=n_min_protein_overlap
    )
    idx_proteins = np.intersect1d(idx_proteins_main_pilot, idx_proteins_model_pilot)

    #overlap_mask = ~np.isnan(x_main) & ~np.isnan(x_pilot)
    #x_main[~overlap_mask] = np.nan
    #x_pilot[~overlap_mask] = np.nan
    #x_est[~overlap_mask] = np.nan

    main_pilot_protein_comparison = metrics.compare_intensities_protein_wise(
        x_main, x_pilot, idx_proteins=idx_proteins, metrics=metric_types, n_min_overlap=n_min_protein_overlap
    )
    model_pilot_protein_comparison = metrics.compare_intensities_protein_wise(
        x_est, x_pilot, idx_proteins=idx_proteins, metrics=metric_types, n_min_overlap=n_min_protein_overlap
    )

    n_metrics = len(metric_types)
    fig, axes = plt.subplots(ncols=n_metrics, figsize=(n_metrics * 6, 5.5))

    for i, metric in enumerate(metric_types):
        ax = axes[i]

        ax.scatter(
            main_pilot_protein_comparison[metric],
            model_pilot_protein_comparison[metric],
            color="blue",
            edgecolor="black",
            linewidth=0,
            s=6,
            alpha=0.5,
        )
        ax.set_xlabel(f"protein-wise {metric} between main and pilot")
        ax.set_ylabel(f"protein-wise {metric} between {model_name} and pilot")
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f"{metric} per protein")


def scatter_pilot_model_by_protein(x_main, x_pilot, x_est):
    miss_mask = np.logical_and(np.isnan(x_main), ~np.isnan(x_pilot))
    protein_mask = miss_mask.any(axis=0)

    x_est_miss_sub = x_est.copy()
    x_est_miss_sub[~miss_mask] = np.nan
    x_est_miss_sub = x_est_miss_sub[:, protein_mask]

    x_pilot_sub = x_pilot.copy()
    x_pilot_sub[~miss_mask] = np.nan
    x_pilot_sub = x_pilot_sub[:, protein_mask]

    x_est_protein = np.nanmean(x_est_miss_sub, axis=0)
    x_pilot_protein = np.nanmean(x_pilot_sub, axis=0)

    pl.scatter_compare_protein_missing_intensity(x_pilot_protein, x_est_protein)


def scatter_pilot_main_by_protein(x_main, x_pilot):
    overlap_mask = np.logical_and(~np.isnan(x_main), ~np.isnan(x_pilot))

    protein_mask = overlap_mask.any(axis=0)

    x_main_sub = x_main.copy()
    x_main_sub[~overlap_mask] = np.nan
    x_main_sub = x_main_sub[:, protein_mask]

    x_pilot_sub = x_pilot.copy()
    x_pilot_sub[~overlap_mask] = np.nan
    x_pilot_sub = x_pilot_sub[:, protein_mask]

    x_main_protein = np.nanmean(x_main_sub, axis=0)
    x_pilot_protein = np.nanmean(x_pilot_sub, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    pl.scatter_compare_protein_missing_intensity(x_pilot_protein, x_main_protein, ax=ax)
    ax.set_title("Overlapping protein intensities in main and pilot")
    ax.set_xlabel("Avg. protein intensities in pilot")
    ax.set_ylabel("Avg. protein intensities in main")


def scatter_pilot_pilot_by_protein(x_main, x_pilot):
    miss_mask = np.logical_and(np.isnan(x_main), ~np.isnan(x_pilot))
    obs_mask = np.logical_and(~np.isnan(x_main), ~np.isnan(x_pilot))

    miss_protein_mask = miss_mask.any(axis=0)
    obs_protein_mask = obs_mask.any(axis=0)

    # overlap between the masks
    protein_mask = obs_protein_mask & miss_protein_mask

    x_obs_sub = x_pilot.copy()
    x_obs_sub[~obs_mask] = np.nan
    x_obs_sub = x_obs_sub[:, protein_mask]

    x_miss_sub = x_pilot.copy()
    x_miss_sub[~miss_mask] = np.nan
    x_miss_sub = x_miss_sub[:, protein_mask]

    x_obs_protein = np.nanmean(x_obs_sub, axis=0)
    x_miss_protein = np.nanmean(x_miss_sub, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    pl.scatter_compare_protein_missing_intensity(x_miss_protein, x_obs_protein, ax=ax)
    ax.set_xlabel("Avg. missing protein intensities")
    ax.set_ylabel("Avg. observed protein intensities")
    ax.set_title("Avg. Intensity per protein between observed and missing in pilot")


def scatter_difference_observed_and_missing_by_protein(x_pilot, x_main, x_model):
    miss_mask = np.logical_and(np.isnan(x_main), ~np.isnan(x_pilot))
    obs_mask = np.logical_and(~np.isnan(x_main), ~np.isnan(x_pilot))

    miss_protein_mask = miss_mask.any(axis=0)
    obs_protein_mask = obs_mask.any(axis=0)

    protein_mask = obs_protein_mask & miss_protein_mask

    # pilot
    x_obs_sub = x_pilot.copy()
    x_obs_sub[~obs_mask] = np.nan
    x_obs_sub = x_obs_sub[:, protein_mask]

    x_miss_sub = x_pilot.copy()
    x_miss_sub[~miss_mask] = np.nan
    x_miss_sub = x_miss_sub[:, protein_mask]

    x_obs_protein = np.nanmean(x_obs_sub, axis=0)
    x_miss_protein = np.nanmean(x_miss_sub, axis=0)

    x_diff_pilot = x_obs_protein - x_miss_protein

    # model
    x_obs_sub = x_model.copy()
    x_obs_sub[~obs_mask] = np.nan
    x_obs_sub = x_obs_sub[:, protein_mask]

    x_miss_sub = x_model.copy()
    x_miss_sub[~miss_mask] = np.nan
    x_miss_sub = x_miss_sub[:, protein_mask]

    x_obs_model = np.nanmean(x_obs_sub, axis=0)
    x_miss_model = np.nanmean(x_miss_sub, axis=0)

    x_diff_model = x_obs_model - x_miss_model

    fig, ax = plt.subplots(figsize=(5, 5))
    pl.scatter_compare_protein_missing_intensity(x_diff_pilot, x_diff_model, ax=ax)
    ax.set_xlabel("Avg. protein difference in pilot")
    ax.set_ylabel("Avg. protein difference in model")
    ax.set_title("Avg. difference per protein between observed and missing")


def plot_diagnostic_groups_umap(adata, field="X_umap"):
    annotation = "diagnosis_group_autoimmune"
    categories = adata.obs[annotation].unique()
    categories = categories[categories != "Other"]

    n_elements = len(categories)

    n_cols = 4
    n_rows = n_elements // n_cols
    n_rows += 1 if n_elements % n_cols != 0 else 0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    colors = sns.color_palette("Set1", n_colors=n_elements)

    for i, group in enumerate(categories):
        mask = adata.obs[annotation] == group

        ax = axes[i // n_cols, i % n_cols]

        umap = adata.obsm[field][~mask]

        ax.scatter(
            umap[:, 0],
            umap[:, 1],
            c = "lightgrey",
            s=10, 
        )

        umap = adata.obsm[field][mask]
        ax.scatter(
            umap[:, 0],
            umap[:, 1],
            c = colors[i],
            s=10, 
        )

        ax.set_title(group, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    for i in range(n_elements, n_cols * n_rows):
        axes[i // n_cols, i % n_cols].axis("off")




def classify_diagnostic_group(adata, field="X_pca", n_components=10):
    X = adata.obsm[field]
    y = adata.obs["Diagnosis_group"].values

    n_components = min(n_components, X.shape[1])
    X = X[:, :n_components]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_test, y_pred
