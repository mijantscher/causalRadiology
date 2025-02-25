import ast
import numpy as np
import pandas as pd
import configparser
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from streamlit_dashboard.causal_cluster_helper import DataLoader, FilterHelper, HypothesisTestHelper
# from helper.psmpy.plotting import PsmPy
from helper.psmpy.psmpy import PsmPy
from typing import TypedDict


class FollowUpFilter(TypedDict):
    UNIQUE_HADM_ID: bool
    MIN_EXAMS_DURING_UNIQUE_HADM_ID: int | None


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
    def __init__(self, data, patient_characteristics, indication_cuis, custom_confounders: list,
                 treatment_var="is_clinical_question_stated", spec_cq_cui=None | str, follow_up_filter=FollowUpFilter | None):
        self.df_matched_treatment = None
        self.df_matched_control = None
        self.data = data
        self.indication_history_cuis = indication_cuis
        self.patient_characteristics = patient_characteristics
        self.data_filtered = None
        self.data_psm = None
        self.psm: PsmPy | None = None
        self.treatment_var = treatment_var if spec_cq_cui is None else "spec_cq_stated"
        self.spec_cq_cui = spec_cq_cui
        self.custom_confounders = custom_confounders
        self.follow_up_filter = follow_up_filter
        if self.follow_up_filter is not None:
            self.custom_confounders += ["is_follow_up"]

    def filter_by_disease_icd_code(self, icd_code: list):
        # self.data_filtered = self.data[self.data["icd_code"].str.contains(f"^{icd_code}")].reset_index(
        #     drop=True)
        query_ = "^" + "|^".join([x[0] for x in icd_code])
        self.data_filtered = self.data[self.data["icd_code"].str.contains(query_)].reset_index(
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
        self.data_filtered.loc[:, "spec_cq_stated"] = 0

    def filter_by_clinical_question(self):
        # query_ = "clinical_q_cuis=='C0032285'"  # C0032285 -> "Pneumonia"
        query_ = f"clinical_q_cuis=='{self.spec_cq_cui}'"
        print("clinical_q_cuis filter query: [", query_, "]")
        tmp = self.data_filtered.explode("clinical_q_cuis")
        tmp_q = tmp.query(query_)
        self.data_filtered.loc[self.data_filtered.index.isin(tmp_q.index), "spec_cq_stated"] = 1

    def add_is_comparison_stated_column(self):
        self.data_filtered["is_comparison_stated"] = FilterHelper.comparison_filter_inplace_column(self.data_filtered)

    def indication_cuis_to_dummies(self, apply_cond_filter):
        # based on: https://saturncloud.io/blog/how-to-convert-a-column-of-list-to-dummies-in-pandas/#:~:text=Converting%20a%20column%20of%20lists%20into%20dummy%20variables%20is%20a,machine%20learning%20and%20statistical%20modeling.
        tt = self.data_filtered["indication_history_cuis"].str.join("|").str.get_dummies()
        if self.spec_cq_cui is None:
            # columns_to_keep = ["is_clinical_question_stated", "indication_history_section_present",
            #                    "is_min_one_indication_history_cuis_in_top_k",
            #                    "is_comparison_stated"] + self.indication_history_cuis + [col for col in
            #                                                                              self.data_filtered.columns if
            #                                                                              '_is_stated' in col]
            columns_to_keep = (
                    ["is_clinical_question_stated"] + self.custom_confounders + self.indication_history_cuis +
                    [col for col in
                     self.data_filtered.columns if
                     '_is_stated' in col])
        else:
            # columns_to_keep = ["spec_cq_stated", "indication_history_section_present",
            #                    "is_min_one_indication_history_cuis_in_top_k",
            #                    "is_comparison_stated"] + self.indication_history_cuis + [col for col in
            #                                                                              self.data_filtered.columns if
            #                                                                              '_is_stated' in col]
            columns_to_keep = (["spec_cq_stated"] + self.custom_confounders + self.indication_history_cuis +
                               [col for col in
                                self.data_filtered.columns if
                                '_is_stated' in col])
        self.data_psm = pd.concat([self.data_filtered, tt], axis=1)
        self.data_psm = self.data_psm[columns_to_keep]
        if apply_cond_filter:
            self.data_psm = self.data_psm[self.data_psm["is_min_one_indication_history_cuis_in_top_k"] == 1]
        self.data_psm.reset_index(inplace=True)

    def patient_characteristics_to_dummies(self):
        dummies_gender = pd.get_dummies(self.data_filtered["gender"])
        dummies_age = pd.get_dummies(self.data_filtered["age_hospitalisation_disc"])
        self.data_psm = pd.concat([self.data_psm, dummies_gender, dummies_age], axis=1)
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

    def psmpy_prop_score_log_reg(self, exclude_vars=None):
        if exclude_vars is None:
            exclude_vars = []
        if self.spec_cq_cui is None:
            df_psm = self.data_psm[["index", "is_clinical_question_stated"] + self.custom_confounders +
                                   self.indication_history_cuis + self.patient_characteristics]
        else:
            df_psm = self.data_psm[["index", "spec_cq_stated"] + self.custom_confounders +
                                   self.indication_history_cuis + self.patient_characteristics]
        print(df_psm.columns)
        self.psm = PsmPy(df_psm, treatment=self.treatment_var, indx='index', exclude=exclude_vars)
        # same as my code using balance=False
        self.psm.logistic_ps(balance=False)

    def psmpy_prop_score_nn(self, model: MLPClassifier, exclude_vars=None):
        if exclude_vars is None:
            exclude_vars = []
        if self.spec_cq_cui is None:
            df_psm = self.data_psm[["index", "is_clinical_question_stated"] + self.custom_confounders +
                                   self.indication_history_cuis + self.patient_characteristics]
        else:
            df_psm = self.data_psm[["index", "spec_cq_stated"] + self.custom_confounders +
                                   self.indication_history_cuis + self.patient_characteristics]
        self.psm = PsmPy(df_psm, treatment=self.treatment_var, indx='index', exclude=exclude_vars)
        # same as my code using balance=False
        self.psm.logistic_nn(logistic=model, balance=False)

    def _psmpy_knn_one_one(self, matcher, replacement, caliper):
        self.psm.knn_matched(matcher=matcher, replacement=replacement, caliper=caliper)
        self.data_psm['matched_psmpy'] = np.nan
        for idx, row in self.psm.matched_ids.iterrows():
            self.data_psm.loc[row["index"], 'matched_psmpy'] = row["matched_ID"]

        # control have no match
        treatment_matched = self.data_psm.dropna(subset=['matched_psmpy'])  # drop not matched

        # matched control observation indexes
        # control_matched_idx = treatment_matched.matched
        control_matched_idx = treatment_matched.matched_psmpy
        control_matched_idx = control_matched_idx.astype(int)  # change to int
        return treatment_matched, self.data_psm.loc[control_matched_idx, :]  # select matched control observations

    def _psmpy_knn_one_many(self, matcher, how_many):
        self.psm.knn_matched_12n(matcher=matcher, how_many=how_many)
        self.data_psm['matched_psmpy'] = np.nan
        major_list_cols = []
        for many in range(how_many):
            col_name = 'largerclass_' + str(many) + 'group'
            major_list_cols.append(col_name)
        for idx, row in self.psm.matched_ids.iterrows():
            self.data_psm.loc[row["index"], 'matched_psmpy'] = str(list(row[major_list_cols]))
        self.data_psm["matched_psmpy"] = self.data_psm["matched_psmpy"].apply(
            lambda e: list(ast.literal_eval(e)) if type(e) == str else [])
        df_tmp = self.data_psm.explode("matched_psmpy")
        df_tmp.reset_index(drop=True, inplace=True)

        # control have no match
        treatment_matched = df_tmp.dropna(subset=['matched_psmpy'])  # drop not matched

        # matched control observation indexes
        control_matched_idx = treatment_matched.matched_psmpy
        control_matched_idx = control_matched_idx.astype(int)  # change to int
        control_matched = df_tmp[df_tmp["index"].isin(control_matched_idx)]  # select matched control observations, this is the old "index" from prior exploding!!

        return treatment_matched, control_matched

    def psmpy_knn(self, matcher='propensity_score', replacement=False, caliper=None, how_many=1):
        if how_many == 1:
            print("knn_matched_one_one(matcher=matcher, replacement=replacement, caliper=caliper)")
            treatment_matched, control_matched = self._psmpy_knn_one_one(matcher, replacement, caliper)
        else:
            print("knn_matched_one_many(matcher=matcher, replacement=replacement, caliper=caliper)")
            treatment_matched, control_matched = self._psmpy_knn_one_many(matcher, how_many)

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

    def filter_follow_up_examinations(self):
        # test hadm_id: 22974590 -> 13 exams
        if self.follow_up_filter["UNIQUE_HADM_ID"]:
            print("follow-up filter applied")
            vc = self.data_filtered["hadm_id"].value_counts()
            vc = vc[vc >= self.follow_up_filter[
                "MIN_EXAMS_DURING_UNIQUE_HADM_ID"]]  # only include samples where more than n studies are performed during same hospital admission

            def _apply_follow_up(e):
                tmp = e.sort_values("datetime_start").reset_index(drop=True)
                tmp['is_follow_up'] = True
                tmp.at[0, 'is_follow_up'] = False
                return tmp

            self.data_filtered = self.data_filtered.groupby("hadm_id").apply(lambda x: _apply_follow_up(x)).reset_index(drop=True)
            # self.data_filtered["is_follow_up"] = self.data_filtered.apply(lambda row: row["hadm_id"] in vc.index, axis=1)
        else:
            print("No follow-up filter applied")

    def generate_psm_data(self, icd_code, apply_cond_filter=False):
        self.filter_by_disease_icd_code(icd_code)
        self.filter_by_indication_history_cuis()
        if self.spec_cq_cui is not None:
            self.filter_by_clinical_question()
        self.add_is_comparison_stated_column()
        if self.follow_up_filter is not None:
            self.filter_follow_up_examinations()
        self.indication_cuis_to_dummies(apply_cond_filter)
        print(self.data_filtered.shape)
        self.patient_characteristics_to_dummies()

    def psm_stats_table_nn_hyperparam_search(self):
        features = self.psm.df_matched.columns.tolist()
        features.remove("index")
        features.remove(self.treatment_var)
        features.remove("propensity_score")
        features.remove("propensity_logit")
        if "matched_ID" in features:
            features.remove("matched_ID")
        agg_operations = {self.treatment_var: 'count'}
        agg_operations.update({
            feature: ['mean', 'std'] for feature in features
        })
        df_stats_matched = self.psm.df_matched.groupby(self.treatment_var).agg(agg_operations)

        agg_operations = {self.treatment_var: 'count'}
        agg_operations.update({
            feature: ['mean', 'std'] for feature in features
        })
        df_stats_origin = self.data_psm.groupby(self.treatment_var).agg(agg_operations)

        # based on the formula from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/
        # and on tutorial: https://ehsanx.github.io/psw/balance.html
        def compute_table_one_smd_dichotomous_vars(table_one: pd.DataFrame, round_digits: int = 4) -> pd.DataFrame:
            feature_smds = []
            for feature in features:
                feature_table_one = table_one[feature].values
                neg_mean = feature_table_one[0, 0]
                pos_mean = feature_table_one[1, 0]

                smd = (pos_mean - neg_mean) / np.sqrt((pos_mean * (1 - pos_mean) + neg_mean * (1 - neg_mean)) / 2)
                smd = round(abs(smd), round_digits)
                feature_smds.append(smd)

            return pd.DataFrame({'features': features, 'smd': feature_smds})

        df_stats_matched_smd_dich = compute_table_one_smd_dichotomous_vars(df_stats_matched)
        df_stats_origin_smd_dich = compute_table_one_smd_dichotomous_vars(df_stats_origin)

        # print("(Matched)  SMD describe")
        # print(df_stats_matched_smd_dich["smd"].describe())
        return df_stats_matched_smd_dich["smd"]

    def psm_stats_table(self):
        features = self.psm.df_matched.columns.tolist()
        features.remove("index")
        features.remove(self.treatment_var)
        features.remove("propensity_score")
        features.remove("propensity_logit")
        if "matched_ID" in features:
            features.remove("matched_ID")
        print(features)
        agg_operations = {self.treatment_var: 'count'}
        agg_operations.update({
            feature: ['mean', 'std'] for feature in features
        })
        df_stats_matched = self.psm.df_matched.groupby(self.treatment_var).agg(agg_operations)
        print(df_stats_matched)

        agg_operations = {self.treatment_var: 'count'}
        agg_operations.update({
            feature: ['mean', 'std'] for feature in features
        })
        df_stats_origin = self.data_psm.groupby(self.treatment_var).agg(agg_operations)
        print(df_stats_origin)

        # based on the formula from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/
        # and on tutorial: https://ehsanx.github.io/psw/balance.html
        def compute_table_one_smd_dichotomous_vars(table_one: pd.DataFrame, round_digits: int = 4) -> pd.DataFrame:
            feature_smds = []
            for feature in features:
                feature_table_one = table_one[feature].values
                neg_mean = feature_table_one[0, 0]
                pos_mean = feature_table_one[1, 0]

                smd = (pos_mean - neg_mean) / np.sqrt((pos_mean * (1 - pos_mean) + neg_mean * (1 - neg_mean)) / 2)
                smd = round(abs(smd), round_digits)
                feature_smds.append(smd)

            return pd.DataFrame({'features': features, 'smd': feature_smds})

        df_stats_matched_smd_dich = compute_table_one_smd_dichotomous_vars(df_stats_matched)
        df_stats_origin_smd_dich = compute_table_one_smd_dichotomous_vars(df_stats_origin)

        print("(Original) Treatment group: ",
              self.data_filtered[self.data_filtered[self.treatment_var] == 1].shape)
        print("(Original) Control group  : ",
              self.data_filtered[self.data_filtered[self.treatment_var] == 0].shape)
        print("(Matched)  Treatment group: ",
              self.psm.df_matched[self.psm.df_matched[self.psm.treatment] == 1].shape)
        print("(Matched)  Control group  : ",
              self.psm.df_matched[self.psm.df_matched[self.psm.treatment] == 0].shape)
        print("(Origin) SMD")
        print(df_stats_origin_smd_dich)
        print("(Matched) SMD")
        print(df_stats_matched_smd_dich)
        print("(Original) SMD describe")
        print(df_stats_origin_smd_dich["smd"].describe())
        print("(Matched)  SMD describe")
        print(df_stats_matched_smd_dich["smd"].describe())


def calculate_causal_effect(tab_, df_data, disease_icd, treatment_var, observation_indication_history,
                            multi_run_estimation=False):
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

    patient_characteristics = [
        "age_intervall_0",
        "age_intervall_1",
        "age_intervall_2",
        "M",
        "F"
    ]

    # disease_icd = "486"  # pneumonia
    disease_icd = [("486", "ICD-9"),
                   ("J189", "ICD-10")]
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
    # psm_helper = PsmHelper(df_data, patient_characteristics, observation_indication_history_cuis)
    # psm_helper.generate_psm_data(disease_icd)

    treatment_var = "is_clinical_question_stated"
    custom_confounders = []
    follow_up_filter: FollowUpFilter = {"UNIQUE_HADM_ID": True,
                                        "MIN_EXAMS_DURING_UNIQUE_HADM_ID": 5}

    psm_helper = PsmHelper(data=df_data,
                           patient_characteristics=patient_characteristics,
                           indication_cuis=observation_indication_history_cuis,
                           custom_confounders=custom_confounders,
                           treatment_var=treatment_var,
                           spec_cq_cui="C0032326",
                           follow_up_filter=follow_up_filter)
    psm_helper.generate_psm_data(disease_icd,
                                 apply_cond_filter=False)
    psm_helper.psmpy_prop_score_log_reg()
    # a matching ration of 1:n only works if there are at least >n*samples_minor_category in the major category!!
    psm_helper.psmpy_knn(matcher="propensity_score", replacement=False, caliper=None, how_many=5)
    print(psm_helper.data_psm)

    psm_helper.psm_stats_table()


if __name__ == '__main__':
    main()
