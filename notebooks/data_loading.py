import pandas as pd
import numpy as np
import scanpy as sc


class DataLoader:

    def __init__(self, data_path):
        self._load_data(data_path=data_path)
        self.adata = self.adata_raw.copy()
        self.adata.raw = self.adata
        self._remove_qc_samples()

    def _filter(self):
        # Filtering from Christine:
        self.adata = self.adata[self.adata.obs['Qalb'] == self.adata.obs['Qalb']]
        self.adata = self.adata[[e not in ['++', '+++', 'bloody'] for e in self.adata.obs['Erythrocytes']]]
        self.adata = self.adata[(self.adata.obs[['Erythrocytes']] == self.adata.obs[['Erythrocytes']]).values]

        self.adata.var['filter'] = 1
        for g in np.unique(self.adata.obs['Diagnosis_group']):
            sub = self.adata[self.adata.obs['Diagnosis_group'] == g]
            completeness = (sub.X == sub.X).mean(axis=0)
            self.adata.var['filter'] = self.adata.var['filter'] * (completeness < 0.2).astype(int)
        self.adata.var['filter'] = self.adata.var['filter'].astype(bool)
        self.adata = self.adata[:, ~self.adata.var['filter']].copy()

    def preprocess_data(
            self,
            filter_cells=0,
            preprocessing_steps={},
    ):
        print(self.adata.shape)
        sc.pp.filter_genes(self.adata, min_cells=1)
        print(self.adata.shape)

        sc.pp.filter_cells(self.adata, min_genes=filter_cells)
        print(self.adata.shape)
        self._filter()
        print(self.adata.shape)
        self.missing_indices = np.where(np.isnan(self.adata.X))

        for step in preprocessing_steps.keys():
            getattr(self, f'_{step}')(**preprocessing_steps[step])

        self.adata.layers['unimputed'] = self.adata.X.copy()
        self.adata.layers['unimputed'][self.missing_indices] = 0

    def _load_data(self, data_path):
        file_name = 'DA-F08.4_-SEC-pass_v06Sc_ion_LibPGQVal1perc_precdLFQdefFull_prot_preprSc03.tsv'
        print(f'loading {file_name}')
        data = pd.read_table(
            f'{data_path}{file_name}',
            index_col='protein',
        )
        data.drop('Unnamed: 0', axis=1, inplace=True)
        var_cols = [c for c in data.columns if 'JaBa' not in c]
        vars = data[var_cols]
        data.drop(var_cols, axis=1, inplace=True)

        adata_raw = sc.AnnData(data.T, var=vars)

        plate = [fn.split('PLATE')[1].split('_')[0] for fn in adata_raw.obs_names]
        adata_raw = adata_raw[[pl[:2] not in ['LT', 'PO', 'QC'] for pl in plate]]
        print(adata_raw.shape)

        row = [fn.split('POS')[1][1] for fn in adata_raw.obs_names]
        col = [str(int(fn.split('POS')[1][2:4])) for fn in adata_raw.obs_names]
        adata_raw.obs['ID'] = [f'{p[:2]}_{r}_{int(c):02d}' for p, r, c in zip(plate, row, col)]
        adata_raw.obs['file'] = adata_raw.obs_names

        drop = ['Comment', 'CVsampleInfo_I', 'CVsampleInfo_II', 'CVsampleInfo_III',
                'QC_Experiment', 'QC_Sample_ID', 'QC_Patient_ID', 'QC_Condition',
                'QC_Condition_numeric', 'PatientID_LT', 'N_puncture_LT',
                'Diff_to_first_puncture_LT', 'ID_MAIN_LT']
        obs = pd.read_table(f'{data_path}annotations_main_lt_v17_Sc07.tsv')
        obs = pd.merge(adata_raw.obs, obs, how='inner', on='ID').drop(drop, axis=1)
        obs = obs.set_index('file')
        adata_raw = adata_raw[obs.index].copy()
        adata_raw.obs = obs.loc[adata_raw.obs.index.values]

        for o in ['Leukocyte_count', 'Albumin_CSF', 'QAlb', 'IgG_CSF', 'QIgG', 'Total_protein']:
            adata_raw.obs[o] = [
                np.nan if a in ['n. best.', 'n. best. ', 'na', 'not measured'] else float(a)
                for a in adata_raw.obs[o]
            ]
        adata_raw.obs['Erythrocytes_in_CSF'] = [
            np.nan if a in ['n. best.', 'n. best. ', 'na', 'not measured'] else a
            for a in adata_raw.obs['Erythrocytes_in_CSF']
        ]
        adata_raw.obs.rename({
            'Leukocyte_count': 'Leukocyte count',
            'Total_protein': 'Total Protein',
            'IgG_CSF': 'IgG CSF',
            'QAlb': 'Qalb',
            'Albumin_CSF': 'Albumin CSF',
            'Erythrocytes_in_CSF': 'Erythrocytes',
            'Sample_plate': 'Platte',
            'Sample_preparation_batch': 'prep_day',
        }, axis=1, inplace=True)

        adata_raw.obs['Total Protein'][adata_raw.obs['Total Protein'] == 0] = np.nan
        adata_raw.obs['Diagnosis_group_subtype'][adata_raw.obs['Diagnosis_group_subtype'] == 'unknown'] = np.nan

        adata_raw.obs['Evosept'] = [a.split('_')[4][1] for a in adata_raw.obs_names]
        adata_raw.obs['Column'] = [a.split('_')[4][3] for a in adata_raw.obs_names]
        adata_raw.obs['Emitter'] = [a.split('_')[4][5] for a in adata_raw.obs_names]
        adata_raw.obs['Capillary'] = [a.split('_')[4][7] for a in adata_raw.obs_names]
        adata_raw.obs['Maintenance'] = [a.split('_')[4][9:11] for a in adata_raw.obs_names]

        adata_raw.obs['log Qalb'] = np.log(adata_raw.obs['Qalb'])

        adata_raw.strings_to_categoricals()

        self.adata_raw = adata_raw

    def _remove_qc_samples(self):
        nr_obs_before = self.adata.shape[0]
        self.adata = self.adata[['ool' not in obs for obs in self.adata.obs['MSgroup']]]
        nr_obs_after = self.adata.shape[0]
        print('Removed %i QCpool samples from the data!' % (nr_obs_before - nr_obs_after))
        self.adata.obs['Age'] = self.adata.obs['Age'].astype('float')

    def _impute_downshifted_normal_sample(
            self,
            scale=0.3,
            shift=1.8,
    ):
        self.adata.X[self.missing_indices] = np.nan
        mean = np.nanmean(self.adata.X, axis=1)
        std = np.nanstd(self.adata.X, axis=1)
        mean_shifted = mean - shift * std
        std_shifted = scale * std
        np.random.seed(42)
        m = np.take(mean_shifted, self.missing_indices[0])
        s = np.take(std_shifted, self.missing_indices[0])
        draws = np.random.normal(m, s)
        self.adata.X[self.missing_indices] = draws

    def _combat(self, obs_key='Platte'):
        sc.pp.combat(self.adata, key=obs_key)
