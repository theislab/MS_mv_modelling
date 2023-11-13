import pandas as pd
import numpy as np
import scanpy as sc
import os
from scipy.stats import spearmanr, pearsonr

import scp.utils as utils

"""
    Functions to load and preprocess the dataset from Mann's lab.
"""


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

    adata = adata[~adata.obs["Qalb"].isna()]
    adata = adata[[e not in ["++", "+++", "bloody"] for e in adata.obs["Erythrocytes"]]]
    adata = adata[(adata.obs[["Erythrocytes"]] == adata.obs[["Erythrocytes"]]).values]

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

    adata = adata[~adata.obs["Qalb"].isna()]
    adata = adata[[e not in ["++", "+++", "bloody"] for e in adata.obs["Erythrocytes"]]]
    adata = adata[(adata.obs[["Erythrocytes"]] == adata.obs[["Erythrocytes"]]).values]

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
        # @TODO: would it be better not to do this correction and thereby get more nan's in the data?
        std_protein_per_plate[std_protein_per_plate == 0] = std

        adata.X[mask, :] = (
            ((x - mean_protein_per_plate) / std_protein_per_plate) * std
        ) + mean_protein

    # result stored in adata.X


def load_data(
    main_dir: str,
    pilot_dir: str,
    do_batch_correction: bool = False,
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

    adata = main_adata
    adata.layers["main"] = main_adata.X.copy()
    adata.layers["pilot"] = pilot_adata.X.copy()

    if verbose:
        x_combined = utils.fill_if_nan(main_adata.X, pilot_adata.X)

        print()
        print(f"main intensity coverage:     {utils.get_coverage(main_adata.X):.2%}")
        print(f"pilot intensity coverage:    {utils.get_coverage(pilot_adata.X):.2%}")
        print(f"combined intensity coverage: {utils.get_coverage(x_combined):.2%}")

    return adata


def compute_common_metrics(x_main, x_pilot, x_est):
    m_pilot = ~np.isnan(x_pilot)
    m_main = ~np.isnan(x_main)

    ## entry-wise
    
    # mse
    main_model_mse = np.mean((x_main[m_main] - x_est[m_main]) ** 2)
    pilot_model_mse = np.mean((x_pilot[m_pilot] - x_est[m_pilot]) ** 2)

    ## protein-wise
    x_est_main_obs = x_est.copy()
    x_est_main_obs[~m_main] = np.nan

    x_est_pilot_obs = x_est.copy()
    x_est_pilot_obs[~m_pilot] = np.nan

    x_main_protein = np.nanmean(x_main, axis=0)
    x_pilot_protein = np.nanmean(x_pilot, axis=0)

    x_est_main_obs_protein = np.nanmean(x_est_main_obs, axis=0)
    x_est_pilot_obs_protein = np.nanmean(x_est_pilot_obs, axis=0)
    
    # mse
    main_model_protein_mse = np.nanmean((x_main_protein - x_est_main_obs_protein) ** 2)
    pilot_model_protein_mse = np.nanmean((x_pilot_protein - x_est_pilot_obs_protein) ** 2)

    result = {
        # entry-wise
        "main_model_mse": main_model_mse,
        "pilot_model_mse": pilot_model_mse,

        # protein-wise
        "main_model_protein_mse": main_model_protein_mse,
        "pilot_model_protein_mse": pilot_model_protein_mse,
    }

    return result


RESULT_DIR = "../../results/manns_lab_data/"


def save_dict_to_results(dict, filename, results_dir=RESULT_DIR):
    path = os.path.join(results_dir, filename)
    utils.save_dict(dict, path)


def load_dict_from_results(filename, results_dir=RESULT_DIR):
    path = os.path.join(results_dir, filename)
    return utils.load_dict(path)
