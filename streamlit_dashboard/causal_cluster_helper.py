import numpy as np
import pandas as pd
import ast
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import re
from typing import TypedDict


class FollowUpFilter(TypedDict):
    UNIQUE_HADM_ID: bool
    MIN_EXAMS_DURING_UNIQUE_HADM_ID: int | None


class DataLoader:
    @staticmethod
    def load_data(engine, observation_classes):
        print("load_data()")
        sql_query = f"-- select t1.*, t4.gender, t4.anchor_age, t4.anchor_year, t2.icd_code, t3.indic_filtered, t3.indication_cuis, t3.clinical_question, t3.clinical_q_cuis, t3.hist_finished, t3.history_cuis, t3.comparison, t3.comparison_cuis from streamlit_timeline_data t1 left join diagnoses_icd t2 on t1.hadm_id = t2.hadm_id left join referral_information t3 on t1.study_id = t3.study left join patients t4 on t1.subject_id = t4.subject_id where t2.seq_num <= 1 and t2.icd_version = 9;"
        sql_query = f"select t1.*, t4.gender, t4.anchor_age, t4.anchor_year, t2.icd_code, t3.indic_filtered, t3.indication_cuis, t3.clinical_question, t3.clinical_q_cuis, t3.hist_finished, t3.history_cuis, t3.comparison, t3.comparison_cuis from streamlit_timeline_data t1 left join diagnoses_icd t2 on t1.hadm_id = t2.hadm_id left join referral_information_v2 t3 on t1.study_id = t3.study left join patients t4 on t1.subject_id = t4.subject_id where t2.seq_num <= 1;"
        print(sql_query)
        df_data = pd.read_sql(sql_query, engine)
        df_data.replace('None', None, inplace=True)
        df_data.drop_duplicates(["study_id"], keep="first",
                                inplace=True)  # this drops multiple xray studies within a single hospital admission
        # todo
        #   if there is no match to referral information. Seems a bit strange because still there is a report in
        #   mimic_cxr_reports and also in chexpert but no entry in the referral_information.
        df_data = df_data[~df_data["indication_cuis"].isnull()].copy(
            deep=True)
        df_data["datetime_start"] = pd.to_datetime(df_data["datetime_start"])
        df_data["datetime_end"] = pd.to_datetime(df_data["datetime_end"])
        df_data["anchor_year"] = pd.to_datetime(df_data["anchor_year"], format="%Y")
        df_data["age_offset"] = (df_data["datetime_start"] - df_data["anchor_year"]).astype("<m8[Y]")
        df_data["age_hospitalisation"] = (df_data["anchor_age"] + df_data["age_offset"]).astype(int)
        bins = [0, 30, 60, 100]
        labels = ["age_intervall_0", "age_intervall_1", "age_intervall_2"]
        df_data['age_hospitalisation_disc'] = pd.cut(df_data['age_hospitalisation'], bins=bins, labels=labels)

        df_data["indication_cuis"] = df_data["indication_cuis"].apply(
            lambda e: ast.literal_eval(e))
        df_data["clinical_q_cuis"] = df_data["clinical_q_cuis"].apply(
            lambda e: ast.literal_eval(e))
        df_data["history_cuis"] = df_data["history_cuis"].apply(lambda e: ast.literal_eval(e))
        df_data["comparison_cuis"] = df_data["comparison_cuis"].apply(
            lambda e: ast.literal_eval(e))
        df_data["indication_history_cuis"] = df_data.apply(
            lambda e: list(set(e["indication_cuis"] + e["history_cuis"])), axis=1)

        df_data.fillna(value=2, inplace=True)

        for path_class in observation_classes:
            df_data[f"{path_class}_is_stated"] = df_data[path_class].apply(lambda x: 0 if x == 2 else 1)

        return df_data

    @staticmethod
    def filter_follow_up_examinations(df_data: pd.DataFrame, follow_up_filter: FollowUpFilter):
        # test hadm_id: 22974590 -> 13 exams
        df_filter = df_data.copy(deep=True)
        if follow_up_filter["UNIQUE_HADM_ID"]:
            print("follow-up filter applied")
            vc = df_data["hadm_id"].value_counts()
            vc = vc[vc >= follow_up_filter[
                "MIN_EXAMS_DURING_UNIQUE_HADM_ID"]]  # only include samples where more than n studies are performed during same hospital admission
            df_filter = df_filter[df_filter["hadm_id"].isin(vc.index)].copy(deep=True)
            df_filter.sort_values(by="datetime_start", inplace=True)
        else:
            print("No follow-up filter applied")
        return df_filter


class FilterHelper:

    @staticmethod
    def referral_filter_advanced(df_filter: pd.DataFrame, filter_by_cui_list: str):
        query_ = filter_by_cui_list.replace("var", "indication_history_cuis")
        print("referral filter query: [", query_, "]")
        tmp = df_filter.explode("indication_history_cuis")
        tmp_q = tmp.query(query_)
        df_filter = df_filter[df_filter.index.isin(tmp_q.index)]
        return df_filter

    @staticmethod
    def clinical_question_filter_advanced(df_filter: pd.DataFrame, filter_by_cui_list: str):
        query_ = filter_by_cui_list.replace("var", "clinical_q_cuis")
        print("referral filter query: [", query_, "]")
        tmp = df_filter.explode("clinical_q_cuis")
        tmp_q = tmp.query(query_)
        df_filter = df_filter[df_filter.index.isin(tmp_q.index)]
        return df_filter

    @staticmethod
    def negative_referral_filter_1(df_filter_1: pd.DataFrame, df_filter_2: pd.DataFrame):
        return pd.concat([df_filter_1, df_filter_2]).drop_duplicates(subset=["study_id"], keep=False)

    @staticmethod
    def negative_clinical_question_filter_1(df_filter_1: pd.DataFrame, df_filter_2: pd.DataFrame):
        return pd.concat([df_filter_1, df_filter_2]).drop_duplicates(subset=["study_id"], keep=False)

    @staticmethod
    def comparison_filter(df_filter: pd.DataFrame):
        df_filter = df_filter[
            (df_filter["indic_filtered"].str.contains("followup", regex=False, na=False, case=False) |
             df_filter["indic_filtered"].str.contains("follow-up", regex=False, na=False, case=False) |
             (
                     df_filter["indic_filtered"].str.contains("eval", regex=False, na=False, case=False) &
                     df_filter["indic_filtered"].str.contains("interval", regex=False, na=False, case=False)
             ) |
             (
                     df_filter["indic_filtered"].str.contains("interval", regex=False, na=False, case=False) &
                     df_filter["indic_filtered"].str.contains("change", regex=False, na=False, case=False)
             ) |
             (
                     df_filter["indic_filtered"].str.contains("compare", regex=False, na=False, case=False) &
                     df_filter["indic_filtered"].str.contains("prior", regex=False, na=False, case=False)
             ))
            |
            (df_filter["hist_finished"].str.contains("followup", regex=False, na=False, case=False) |
             df_filter["hist_finished"].str.contains("follow-up", regex=False, na=False, case=False) |
             (
                     df_filter["hist_finished"].str.contains("eval", regex=False, na=False, case=False) &
                     df_filter["hist_finished"].str.contains("interval", regex=False, na=False, case=False)
             ) |
             (
                     df_filter["hist_finished"].str.contains("interval", regex=False, na=False, case=False) &
                     df_filter["hist_finished"].str.contains("change", regex=False, na=False, case=False)
             ) |
             (
                     df_filter["hist_finished"].str.contains("compare", regex=False, na=False, case=False) &
                     df_filter["hist_finished"].str.contains("prior", regex=False, na=False, case=False)
             ))
            ]

        return df_filter

    @staticmethod
    def comparison_filter_inplace_column(df_filter: pd.DataFrame):
        def _comparison_apply(row):
            if ((re.search("followup", str(row["indic_filtered"]), re.IGNORECASE) or
                 re.search("follow-up", str(row["indic_filtered"]), re.IGNORECASE) or
                 (
                         re.search("eval", str(row["indic_filtered"]), re.IGNORECASE) and
                         re.search("interval", str(row["indic_filtered"]), re.IGNORECASE)
                 ) or
                 (
                         re.search("interval", str(row["indic_filtered"]), re.IGNORECASE) and
                         re.search("change", str(row["indic_filtered"]), re.IGNORECASE)
                 ) or
                 (
                         re.search("compare", str(row["indic_filtered"]), re.IGNORECASE) and
                         re.search("prior", str(row["indic_filtered"]), re.IGNORECASE)
                 ))
                    or
                    (re.search("followup", str(row["hist_finished"]), re.IGNORECASE) or
                     re.search("follow-up", str(row["hist_finished"]), re.IGNORECASE) or
                     (
                             re.search("eval", str(row["hist_finished"]), re.IGNORECASE) and
                             re.search("interval", str(row["hist_finished"]), re.IGNORECASE)
                     ) or
                     (
                             re.search("interval", str(row["hist_finished"]), re.IGNORECASE) and
                             re.search("change", str(row["hist_finished"]), re.IGNORECASE)
                     ) or
                     (
                             re.search("compare", str(row["hist_finished"]), re.IGNORECASE) and
                             re.search("prior", str(row["hist_finished"]), re.IGNORECASE)
                     ))):
                return 1
            return 0

        return df_filter.apply(_comparison_apply, axis=1)

    @staticmethod
    def comparison_filter_neg(df_filter: pd.DataFrame):
        df_filter = df_filter[
            (~df_filter["indic_filtered"].str.contains("followup", regex=False, na=False, case=False) &
             ~df_filter["indic_filtered"].str.contains("follow-up", regex=False, na=False, case=False) &
             ~(
                     df_filter["indic_filtered"].str.contains("eval", regex=False, na=False, case=False) &
                     df_filter["indic_filtered"].str.contains("interval", regex=False, na=False, case=False)
             ) &
             ~(
                     df_filter["indic_filtered"].str.contains("interval", regex=False, na=False, case=False) &
                     df_filter["indic_filtered"].str.contains("change", regex=False, na=False, case=False)
             ) &
             ~(
                     df_filter["indic_filtered"].str.contains("compare", regex=False, na=False, case=False) &
                     df_filter["indic_filtered"].str.contains("prior", regex=False, na=False, case=False)
             ))
            &
            (~df_filter["hist_finished"].str.contains("followup", regex=False, na=False, case=False) &
             ~df_filter["hist_finished"].str.contains("follow-up", regex=False, na=False, case=False) &
             ~(
                     df_filter["hist_finished"].str.contains("eval", regex=False, na=False, case=False) &
                     df_filter["hist_finished"].str.contains("interval", regex=False, na=False, case=False)
             ) &
             ~(
                     df_filter["hist_finished"].str.contains("interval", regex=False, na=False, case=False) &
                     df_filter["hist_finished"].str.contains("change", regex=False, na=False, case=False)
             ) &
             ~(
                     df_filter["hist_finished"].str.contains("compare", regex=False, na=False, case=False) &
                     df_filter["hist_finished"].str.contains("prior", regex=False, na=False, case=False)
             ))
            ]

        return df_filter


class VisualizationHelper:
    @staticmethod
    def histogram_pathologies(fig, df_cluster, observation_classes, title="[Filter_1] Histogram pathologies"):
        for obs_class in observation_classes:
            fig.add_trace(go.Histogram(x=df_cluster[obs_class], name=obs_class,
                                       histnorm="probability density"))

        fig.update_layout(title=title, barmode='group')
        return fig

    @staticmethod
    def histogram_pathologies_binary(fig, df_cluster, observation_classes, title="[Filter_1] Pathology stated"):
        for path_class in observation_classes:
            fig.add_trace(go.Histogram(x=df_cluster[f"{path_class}_is_stated"], name=f"{path_class}_is_stated",
                                       histnorm="probability density"))

        fig.update_layout(title=title, barmode='group')
        return fig


class HypothesisTestHelper:

    @staticmethod
    def pathologies_chi2_test_binary(df_cluster_1, df_cluster_2, observation_classes) -> pd.DataFrame:
        stats_dict = {
            "pathology": [],
            "chi2": [],
            "p_value": [],
            "dof": [],
            "expected": []
        }
        for path_class in observation_classes:
            cluster_1_counts = df_cluster_1[f"{path_class}_is_stated"]
            cluster_2_counts = df_cluster_2[f"{path_class}_is_stated"]
            path_contingency_table = np.array([[np.sum(cluster_1_counts == 0), np.sum(cluster_1_counts == 1)],
                                               [np.sum(cluster_2_counts == 0), np.sum(cluster_2_counts == 1)]])

            if np.any(path_contingency_table == 0):
                stats_dict["pathology"].append(path_class)
                stats_dict["chi2"].append(np.nan)
                stats_dict["p_value"].append(np.nan)
                stats_dict["dof"].append(np.nan)
                stats_dict["expected"].append(np.nan)
                continue

            chi2, p_value, dof, expected = chi2_contingency(path_contingency_table)
            stats_dict["pathology"].append(path_class)
            stats_dict["chi2"].append(chi2)
            stats_dict["p_value"].append(p_value)
            stats_dict["dof"].append(dof)
            stats_dict["expected"].append(expected)
        return pd.DataFrame(stats_dict)

    @staticmethod
    def pathologies_chi2_test(df_cluster_1, df_cluster_2, observation_classes):
        stats_dict = {
            "pathology": [],
            "chi2": [],
            "p_value": [],
            "dof": [],
            "expected": []
        }
        for path_class in observation_classes:
            cluster_1_counts = df_cluster_1[path_class]
            cluster_2_counts = df_cluster_2[path_class]

            if path_class == "No_Finding":
                path_contingency_table = np.array([[np.sum(cluster_1_counts == 1), np.sum(cluster_1_counts == 2)],
                                                   [np.sum(cluster_2_counts == 1), np.sum(cluster_2_counts == 2)]])
            else:
                path_contingency_table = np.array([[np.sum(cluster_1_counts == -1), np.sum(cluster_1_counts == 0),
                                                    np.sum(cluster_1_counts == 1), np.sum(cluster_1_counts == 2)],
                                                   [np.sum(cluster_2_counts == -1), np.sum(cluster_2_counts == 0),
                                                    np.sum(cluster_2_counts == 1), np.sum(cluster_2_counts == 2)]])
            print("Contingency table for ", path_class, ": ")
            print(path_contingency_table)
            if np.any(path_contingency_table == 0):
                stats_dict["pathology"].append(path_class)
                stats_dict["chi2"].append(np.nan)
                stats_dict["p_value"].append(np.nan)
                stats_dict["dof"].append(np.nan)
                stats_dict["expected"].append(np.nan)
                continue
            chi2, p_value, dof, expected = chi2_contingency(path_contingency_table)
            stats_dict["pathology"].append(path_class)
            stats_dict["chi2"].append(chi2)
            stats_dict["p_value"].append(p_value)
            stats_dict["dof"].append(dof)
            stats_dict["expected"].append(expected)
        return pd.DataFrame(stats_dict)

    @staticmethod
    def chi2_test(observed_freq):
        chi2, p, dof, expected = chi2_contingency(observed_freq)
        return chi2, p, dof, expected
