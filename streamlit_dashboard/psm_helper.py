import numpy as np
import pandas as pd
import configparser
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
# from streamlit_dashboard.causal_cluster_helper import DataLoader, FilterHelper, HypothesisTestHelper
from causal_cluster_helper import DataLoader, FilterHelper, HypothesisTestHelper
from psmpy.plotting import PsmPy

OBSERVATIONAL_CLASSES = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged_Cardiomediastinum',
    'Fracture',
    'Lung_Lesion',
    'Lung_Opacity',
    'No_Finding',
    'Pleural_Effusion',
    'Pleural_Other',
    'Pneumonia',
    'Pneumothorax',
    'Support_Devices'
]


class PsmHelper:
    def __init__(self, data, indication_cuis, treatment_var="is_clinical_question_stated"):
        self.df_matched_treatment = None
        self.df_matched_control = None
        self.data = data
        self.indication_history_cuis = indication_cuis
        self.data_filtered = None
        self.data_psm = None
        self.psm = None
        self.treatment_var = treatment_var

    def filter_by_disease_icd_code(self, icd_code):
        self.data_filtered = self.data[self.data["icd_code"].str.contains(f"^{icd_code}")].reset_index(
            drop=True)

    def filter_by_indication_history_cuis(self):
        def _apply_indication_filter(row):
            return [x for x in row if x in self.indication_history_cuis]  # this is a "|" connection

        self.data_filtered["indication_history_section_present"] = self.data_filtered.apply(
            lambda e: 1 if (e["indic_filtered"] != 2 or e["hist_finished"] != 2) else 0, axis=1)
        self.data_filtered["indication_cuis"] = self.data_filtered["indication_cuis"].apply(
            lambda e: _apply_indication_filter(e))
        self.data_filtered["history_cuis"] = self.data_filtered["history_cuis"].apply(
            lambda e: _apply_indication_filter(e))
        self.data_filtered["indication_history_cuis"] = self.data_filtered["indication_history_cuis"].apply(
            lambda e: _apply_indication_filter(e))
        self.data_filtered["is_min_one_indication_history_cuis_in_top_k"] = self.data_filtered[
            "indication_history_cuis"].apply(
            lambda e: 1 if len(e) > 0 else 0)
        self.data_filtered["is_clinical_question_stated"] = self.data_filtered["clinical_question"].apply(
            lambda e: 0 if e == 2 else 1)

    def add_is_comparison_stated_column(self):
        self.data_filtered["is_comparison_stated"] = FilterHelper.comparison_filter_inplace_column(self.data_filtered)

    def indication_cuis_to_dummies(self, apply_cond_filter):
        # based on: https://saturncloud.io/blog/how-to-convert-a-column-of-list-to-dummies-in-pandas/#:~:text=Converting%20a%20column%20of%20lists%20into%20dummy%20variables%20is%20a,machine%20learning%20and%20statistical%20modeling.
        tt = self.data_filtered["indication_history_cuis"].str.join("|").str.get_dummies()
        columns_to_keep = ["is_clinical_question_stated", "indication_history_section_present",
                           "is_min_one_indication_history_cuis_in_top_k",
                           "is_comparison_stated"] + self.indication_history_cuis + [col for col in
                                                                                     self.data_filtered.columns if
                                                                                     '_is_stated' in col]
        self.data_psm = pd.concat([self.data_filtered, tt], axis=1)
        self.data_psm = self.data_psm[columns_to_keep]
        if apply_cond_filter:
            self.data_psm = self.data_psm[self.data_psm["is_min_one_indication_history_cuis_in_top_k"] == 1]
        self.data_psm.reset_index(inplace=True)

    def sample_matching_custom_implementation(self, neighbor_indexes):
        # for each point in treatment, we find a matching point in control without replacement
        # note the 10 neighbors may include both points in treatment and control

        matched_control = []  # keep track of the matched observations in control

        for current_index, row in self.data_psm.iterrows():  # iterate over the dataframe
            already_matched = False
            if row[self.treatment_var] == 0:  # the current row is in the control group
                self.data_psm.loc[current_index, 'matched'] = np.nan  # set matched to nan
            else:
                for idx in neighbor_indexes[current_index, :]:  # for each row in treatment, find the k neighbors
                    # make sure the current row is not the idx - don't match to itself
                    # and the neighbor is in the control
                    if (current_index != idx) and (self.data_psm.loc[idx][self.treatment_var] == 0):
                        if idx not in matched_control:  # this control has not been matched yet
                            self.data_psm.loc[current_index, 'matched'] = idx  # record the matching
                            matched_control.append(idx)  # add the matched to the list
                            already_matched = True
                            break
                        else:
                            continue
                # When you get here, already all nns are in matched control. So choose one (which replacement)
                if not already_matched:
                    to_select_arr = np.intersect1d(neighbor_indexes[current_index, :], np.array(matched_control))
                    # selected_idx = to_select_arr[np.random.choice(len(to_select_arr), size=1, replace=True)][0]
                    selected_idx = to_select_arr[0]
                    self.data_psm.loc[current_index, 'matched'] = selected_idx  # record the matching
        return matched_control

    def psmpy_prop_score(self, exclude_vars=None):
        if exclude_vars is None:
            exclude_vars = []
        df_psm = self.data_psm[["index", "is_clinical_question_stated", "is_comparison_stated",
                                "indication_history_section_present"] + self.indication_history_cuis]
        self.psm = PsmPy(df_psm, treatment=self.treatment_var, indx='index', exclude=exclude_vars)
        # same as my code using balance=False
        self.psm.logistic_ps(balance=False)

    def psmpy_knn(self, matcher='propensity_score', replacement=False):
        self.psm.knn_matched(matcher=matcher, replacement=replacement, caliper=None)
        self.data_psm['matched_psmpy'] = np.nan

        for idx, row in self.psm.matched_ids.iterrows():
            self.data_psm.loc[row["index"], 'matched_psmpy'] = row["matched_ID"]

        # control have no match
        treatment_matched = self.data_psm.dropna(subset=['matched_psmpy'])  # drop not matched

        # matched control observation indexes
        # control_matched_idx = treatment_matched.matched
        control_matched_idx = treatment_matched.matched_psmpy
        control_matched_idx = control_matched_idx.astype(int)  # change to int
        control_matched = self.data_psm.loc[control_matched_idx, :]  # select matched control observations

        # combine the matched treatment and control
        df_matched = pd.concat([treatment_matched, control_matched])

        df_matched[self.treatment_var].value_counts()

        # matched control and treatment
        self.df_matched_control = df_matched[df_matched[self.treatment_var] == 0]
        self.df_matched_treatment = df_matched[df_matched[self.treatment_var] == 1]

    def psmpy_chi2_test_results(self) -> pd.DataFrame:
        # chi2 test for each pathology class (dependent/output variable) WITH matching
        test_stats = HypothesisTestHelper.pathologies_chi2_test_binary(self.df_matched_control,
                                                                       self.df_matched_treatment,
                                                                       OBSERVATIONAL_CLASSES)
        test_stats["is_significant"] = test_stats["p_value"].apply(lambda e: True if e < 0.05 else False)
        return test_stats

    def generate_psm_data(self, icd_code, apply_cond_filter=False):
        self.filter_by_disease_icd_code(icd_code)
        self.filter_by_indication_history_cuis()
        self.add_is_comparison_stated_column()
        self.indication_cuis_to_dummies(apply_cond_filter)


def calculate_causal_effect(tab_, df_data, disease_icd, treatment_var, observation_indication_history, multi_run_estimation=False):
    observation_indication_history_cuis = [x[0] for x in observation_indication_history]
    if not multi_run_estimation:
        psm_helper = PsmHelper(df_data, observation_indication_history_cuis, treatment_var)
        psm_helper.generate_psm_data(disease_icd, apply_cond_filter=False)

        psm_helper.psmpy_prop_score()
        psm_helper.psmpy_knn(matcher="propensity_score", replacement=False)
        test_stats = psm_helper.psmpy_chi2_test_results()
        tab_.write(test_stats)
    else:
        test_stats_dict = {
            "run_id": [],
            "pathology": [],
            "chi2": [],
            "p_value": [],
            "is_significant": []
        }

        for run_id, symptoms in enumerate(observation_indication_history):
            symptoms_list = observation_indication_history_cuis[0:(run_id + 1)]
            print("(", run_id, ")", " Estimation for: ", symptoms_list)

            psm_helper = PsmHelper(df_data, observation_indication_history_cuis[0:(run_id + 1)], treatment_var)
            psm_helper.generate_psm_data(disease_icd, apply_cond_filter=False)
            psm_helper.psmpy_prop_score()
            psm_helper.psmpy_knn(matcher="propensity_score", replacement=False)
            test_stats = psm_helper.psmpy_chi2_test_results()

            for idx, row in test_stats.iterrows():
                test_stats_dict["run_id"].append(run_id)
                test_stats_dict["pathology"].append(row["pathology"])
                test_stats_dict["chi2"].append(row["chi2"])
                test_stats_dict["p_value"].append(row["p_value"])
                test_stats_dict["is_significant"].append(row["is_significant"])
        df_test_stats = pd.DataFrame(test_stats_dict)
        tab_.write(df_test_stats)
        fig_p_value, fig_effect_sign = plot_causal_effect_stats(df_test_stats)
        tab_.plotly_chart(fig_p_value)
        tab_.plotly_chart(fig_effect_sign)


def plot_causal_effect_stats(df_test_stats: pd.DataFrame):
    df_plot_p_value = pd.pivot(df_test_stats[["run_id", "pathology", "p_value"]], index="run_id", columns="pathology",
                               values="p_value")
    df_plot_chi2 = pd.pivot(df_test_stats[["run_id", "pathology", "chi2"]], index="run_id", columns="pathology",
                            values="chi2")
    df_plot_is_significant = pd.pivot(df_test_stats[["run_id", "pathology", "is_significant"]], index="run_id",
                                      columns="pathology", values="is_significant")

    fig_p_value = go.Figure()
    fig_effect_sign = go.Figure()
    fig_p_value = px.line(df_plot_p_value, x=df_plot_p_value.index, y=df_plot_p_value.columns,
                  title="Chi2 p_value effect significance")
    fig_effect_sign = px.line(df_plot_is_significant, x=df_plot_is_significant.index, y=df_plot_is_significant.columns,
                  title="Effect significance")
    return fig_p_value, fig_effect_sign


def main():
    config = configparser.ConfigParser()
    config.read('../../data/config/local_config.ini')
    conn = sqlite3.connect(config['DATABASE']['path'])

    disease_icd = "486"  # pneumonia
    observation_indication_history = [
        ("C0015967", "Fever"),
        ("C0013404", "Dyspnea"),
        ("C0010200", "Coughing"),
        ("C0032285", "Pneumonia"),
        ("C0242184", "Hypoxia"),
        ("C0032227", "Pleural effusion disorder"),
        ("C0008031", "Chest Pain")
    ]
    observation_indication_history_cuis = [x[0] for x in observation_indication_history]

    df_data = DataLoader.load_data(conn, OBSERVATIONAL_CLASSES)
    psm_helper = PsmHelper(df_data, observation_indication_history_cuis)
    psm_helper.generate_psm_data(disease_icd)
    print(psm_helper.data_psm)


if __name__ == '__main__':
    main()
