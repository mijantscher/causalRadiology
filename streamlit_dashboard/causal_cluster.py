import configparser
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from sqlite3 import Connection
from tqdm import tqdm
from PIL import Image
from causal_cluster_helper import FilterHelper, VisualizationHelper, HypothesisTestHelper, DataLoader, FollowUpFilter

tqdm.pandas()

print("Start")
config = configparser.ConfigParser()
config.read('../data/config/local_config.ini')

config_filter = configparser.ConfigParser()
config_filter.read('../data/config/streamlit_constants.ini')

observation_classes = [
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

observation_indication_history = [
    ("C0015967", "Fever"),
    ("C0013404", "Dyspnea"),
    ("C0010200", "Coughing"),
    ("C0032285", "Pneumonia"),
    ("C0242184", "Hypoxia"),
    ("C0032227", "Pleural effusion disorder"),
    ("C0008031", "Chest Pain"),
    ("C0600500", "Peptide Nucleic Acids"),
    ("C0024117", "Chronic Obstructive Airway Disease"),
    ("C0332271", "Worsening pattern"),
    ("C0239134", "Productive Cough"),
    ("C0018802", "Congestive heart failure"),
    ("C0231835", "Tachypnea"),
    ("C0242379", "Malignant neoplasm of lung"),
]


@st.cache(allow_output_mutation=True)
def get_connection(path):
    """Put the connection in cache to reuse if path does not change."""
    print("db path: ", path)
    return sqlite3.connect(path, check_same_thread=False)


@st.cache(hash_funcs={Connection: id})
def load_data(engine):
    return DataLoader.load_data(engine, observation_classes)


@st.cache(hash_funcs={Connection: id})
def load_lookup_data(engine):
    print("load_lookup_data()")
    sql_query = "select * from referral_information_cui_lookup"
    print(sql_query)
    df_lookup_data = pd.read_sql(sql_query, engine)

    return df_lookup_data


conn = get_connection(config['DATABASE']['path'])

# **************************************************************************************
# ******************************* Visualization Header *********************************
# **************************************************************************************
tab_causal_graph, tab_referral_stats, tab_output_stats, tab_tests, tab_causal_effect = st.tabs(
    ["Causal graph", "Referral Stats", "Output Stats", "Hypothesis Tests", "Causal Effect estimation"])
tab_referral_stats_col_1, tab_referral_stats_col_2 = tab_referral_stats.columns([2, 1])

finding_columns = ['Atelectasis', 'Cardiomegaly',
                   'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum', 'Fracture',
                   'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                   'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

hint_str = "Enter the query in the form: var=='C0015967'&(var=='C0013404'|var=='C0010200')"
hint_str_conditions = "Enter the query in the form: Atelectasis_is_stated=1"
icd_code_filter = st.sidebar.text_input('ICD-Code input', '^486')
indication_filter = st.sidebar.text_input('Indication and history input 1', '', help=hint_str)
cq_filter_1 = st.sidebar.text_input('Clinical question input 1', '', help=hint_str,
                                    disabled=False)
condition_filter_1 = st.sidebar.text_input('Condition filter 1', '', help=hint_str_conditions)
comparison_filter = st.sidebar.selectbox(
    "Comparison filter",
    ("No filter", "Comparison section", "No comparison section")
)

st.sidebar.write("---")

icd_code_filter_2 = st.sidebar.text_input('ICD-Code input 2', '^486')
neg_indication_filter_1 = st.sidebar.checkbox('Negate indication input 1')

if neg_indication_filter_1:
    indication_filter_2 = st.sidebar.text_input('Indication and history input 2', '', help=hint_str,
                                                disabled=True)
else:
    indication_filter_2 = st.sidebar.text_input('Indication and history input 2', '', help=hint_str,
                                                disabled=False)

neg_clinical_question_filter_1 = st.sidebar.checkbox('Negate question input 1')

if neg_clinical_question_filter_1:
    cq_filter_2 = st.sidebar.text_input('Clinical question input 2', '', help=hint_str,
                                        disabled=True)
else:
    cq_filter_2 = st.sidebar.text_input('Clinical question input 2', '', help=hint_str,
                                        disabled=False)
condition_filter_2 = st.sidebar.text_input('Condition filter 2', '', help=hint_str_conditions)
comparison_filter_2 = st.sidebar.selectbox(
    "Comparison filter 2",
    ("No filter", "Comparison section", "No comparison section")
)

st.sidebar.write("---")
follow_up_filter_applied = st.sidebar.checkbox('Filter by follow-up examinations')
follow_up_filter_min_exams_required = 0
if follow_up_filter_applied:
    follow_up_filter_min_exams_required = st.sidebar.number_input("Insert a number",
                                                                  value=2,
                                                                  step=1)

# **************************************************************************************
# ******************************** Visualization Body **********************************
# **************************************************************************************
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
df_data = load_data(conn)
if follow_up_filter_applied:
    follow_up_filter_: FollowUpFilter = {"UNIQUE_HADM_ID": follow_up_filter_applied,
                                         "MIN_EXAMS_DURING_UNIQUE_HADM_ID": int(follow_up_filter_min_exams_required)}
    df_data = DataLoader.filter_follow_up_examinations(df_data, follow_up_filter_)
df_lookup_data = load_lookup_data(conn)
data_load_state.text('')

tab_output_stats.title('MIMIC-Cluster Dashboard')
df_cluster_1 = df_data[df_data["icd_code"].str.contains(f"{icd_code_filter}")]
df_cluster_2 = df_data[df_data["icd_code"].str.contains(f"{icd_code_filter_2}")]

if condition_filter_1 != "":
    condition_key, condition_val = condition_filter_1.split("=")
    df_cluster_1 = df_cluster_1[df_cluster_1[condition_key] == int(condition_val)]

if condition_filter_2 != "":
    condition_key_2, condition_val_2 = condition_filter_2.split("=")
    df_cluster_2 = df_cluster_2[df_cluster_2[condition_key_2] == int(condition_val_2)]

# ******************************** Indication filter ***********************************
if indication_filter != "":
    print("Filter by indication (and history) filter 1")
    # df_cluster_1 = FilterHelper.referral_filter(df_cluster_1, indication_filter.split("&"))
    df_cluster_1 = FilterHelper.referral_filter_advanced(df_cluster_1, indication_filter)

if neg_indication_filter_1:
    print("Filter by negative filter 1 indication (and history) filter 2")
    df_cluster_2 = FilterHelper.negative_referral_filter_1(df_cluster_1, df_cluster_2)
elif indication_filter_2 != "":
    print("Filter by indication (and history) filter 2")
    df_cluster_2 = FilterHelper.referral_filter_advanced(df_cluster_2, indication_filter_2)
# **************************************************************************************

# ************************ Clinical question filter ************************************
if cq_filter_1 != "":
    print("Filter by clinical question filter 1")
    df_cluster_1 = FilterHelper.clinical_question_filter_advanced(df_cluster_1, cq_filter_1)
if neg_clinical_question_filter_1:
    print("Filter by negative clinical question filter 2")
    print("df_cluster_1.shape: ", df_cluster_1.shape)
    print("df_cluster_2.shape: ", df_cluster_2.shape)
    df_cluster_2 = FilterHelper.negative_clinical_question_filter_1(df_cluster_1, df_cluster_2)
    print("df_cluster_2.shape: ", df_cluster_2.shape)
elif cq_filter_2 != "":
    print("Filter by clinical question filter 2")
    df_cluster_2 = FilterHelper.clinical_question_filter_advanced(df_cluster_2, cq_filter_2)
# **************************************************************************************

# ***************************** Comparison filter **************************************
if comparison_filter == "Comparison section":
    print("Filter by comparison filter 1")
    df_cluster_1 = FilterHelper.comparison_filter(df_cluster_1)
elif comparison_filter == "No comparison section":
    df_cluster_1 = FilterHelper.comparison_filter_neg(df_cluster_1)

if comparison_filter_2 == "Comparison section":
    print("Filter by comparison filter 2")
    df_cluster_2 = FilterHelper.comparison_filter(df_cluster_2)
elif comparison_filter_2 == "No comparison section":
    df_cluster_2 = FilterHelper.comparison_filter_neg(df_cluster_2)
# **************************************************************************************

# *****************************************
tab_output_stats.text(f"[Filter_1] dataframe shape: {df_cluster_1.shape}")
tab_output_stats.text(f"[Filter_2] dataframe shape: {df_cluster_2.shape}")
# *****************************************
fig = VisualizationHelper.histogram_pathologies(go.Figure(), df_cluster_1, observation_classes,
                                                "[Filter_1] Histogram pathologies")
tab_output_stats.plotly_chart(fig)

fig_2 = VisualizationHelper.histogram_pathologies(go.Figure(), df_cluster_2, observation_classes,
                                                  "[Filter_2] Histogram pathologies")
tab_output_stats.plotly_chart(fig_2)
# *****************************************

# *****************************************
fig_1 = VisualizationHelper.histogram_pathologies_binary(go.Figure(), df_cluster_1, observation_classes,
                                                         "[Filter_1] Pathology stated")
tab_output_stats.plotly_chart(fig_1)

fig_3 = VisualizationHelper.histogram_pathologies_binary(go.Figure(), df_cluster_2, observation_classes,
                                                         "[Filter_2] Pathology stated")
tab_output_stats.plotly_chart(fig_3)
# *****************************************

NR_BINS_HIST = 100
tab_referral_stats.text(f"[Filter_1] dataframe shape: {df_cluster_1.shape}")
tab_referral_stats.text(f"[Filter_2] dataframe shape: {df_cluster_2.shape}")

tab_referral_stats_col_1.title('Indication and history stats')
df_exploded_cluster_1 = df_cluster_1.explode("indication_history_cuis")
df_referral_stats_cluster_1 = df_exploded_cluster_1["indication_history_cuis"].value_counts().iloc[:NR_BINS_HIST]
df_hist_lookup_cluster_1 = pd.merge(df_referral_stats_cluster_1, df_lookup_data, how="left", left_index=True,
                                    right_on="CUI")
df_hist_lookup_cluster_1 = df_hist_lookup_cluster_1.reset_index(drop=True)
df_hist_lookup_cluster_1.rename(columns={"indication_history_cuis": "count"}, inplace=True)

tab_referral_stats_col_1.write(df_hist_lookup_cluster_1[["CUI", "Name", "count"]])

fig_4 = go.Figure(go.Bar(x=df_referral_stats_cluster_1.index, y=df_referral_stats_cluster_1.values))
fig_4.update_layout(title="[Filter 1] CUI distribution indication")

tab_referral_stats_col_2.plotly_chart(fig_4)

df_exploded_cluster_2 = df_cluster_2.explode("indication_history_cuis")
df_referral_stats_cluster_2 = df_exploded_cluster_2["indication_history_cuis"].value_counts().iloc[:NR_BINS_HIST]
df_hist_lookup_cluster_2 = pd.merge(df_referral_stats_cluster_2, df_lookup_data, how="left", left_index=True,
                                    right_on="CUI")
df_hist_lookup_cluster_2 = df_hist_lookup_cluster_2.reset_index(drop=True)
df_hist_lookup_cluster_2.rename(columns={"indication_history_cuis": "count"}, inplace=True)

tab_referral_stats_col_1.write(df_hist_lookup_cluster_2[["CUI", "Name", "count"]])

fig_5 = go.Figure(go.Bar(x=df_referral_stats_cluster_2.index, y=df_referral_stats_cluster_2.values))
fig_5.update_layout(title="[Filter 2] CUI distribution indication")

tab_referral_stats_col_2.plotly_chart(fig_5)
tab_referral_stats_col_1.write("---")
# ------------------------------------------------------------------------------------------------------------

tab_referral_stats_col_1.title('Clinical question stats')
df_exploded_cluster_1 = df_cluster_1.explode("clinical_q_cuis")
df_referral_stats_cluster_1 = df_exploded_cluster_1["clinical_q_cuis"].value_counts().iloc[:NR_BINS_HIST]
df_hist_lookup_cluster_1 = pd.merge(df_referral_stats_cluster_1, df_lookup_data, how="left", left_index=True,
                                    right_on="CUI")
df_hist_lookup_cluster_1 = df_hist_lookup_cluster_1.reset_index(drop=True)
df_hist_lookup_cluster_1.rename(columns={"clinical_q_cuis": "count"}, inplace=True)

tab_referral_stats_col_1.write(df_hist_lookup_cluster_1[["CUI", "Name", "count"]])

fig_6 = go.Figure(go.Bar(x=df_referral_stats_cluster_1.index, y=df_referral_stats_cluster_1.values))
fig_6.update_layout(title="[Filter 1] CUI distribution clinical question")

tab_referral_stats_col_2.plotly_chart(fig_6)

df_exploded_cluster_2 = df_cluster_2.explode("clinical_q_cuis")
df_referral_stats_cluster_2 = df_exploded_cluster_2["clinical_q_cuis"].value_counts().iloc[:NR_BINS_HIST]
df_hist_lookup_cluster_2 = pd.merge(df_referral_stats_cluster_2, df_lookup_data, how="left", left_index=True,
                                    right_on="CUI")
df_hist_lookup_cluster_2 = df_hist_lookup_cluster_2.reset_index(drop=True)
df_hist_lookup_cluster_2.rename(columns={"clinical_q_cuis": "count"}, inplace=True)

tab_referral_stats_col_1.write(df_hist_lookup_cluster_2[["CUI", "Name", "count"]])

fig_7 = go.Figure(go.Bar(x=df_referral_stats_cluster_2.index, y=df_referral_stats_cluster_2.values))
fig_7.update_layout(title="[Filter 2] CUI distribution clinical question")

tab_referral_stats_col_2.plotly_chart(fig_7)
tab_referral_stats_col_1.write("---")
# ------------------------------------------------------------------------------------------------------------

tab_referral_stats_col_1.title('History stats')
df_exploded_cluster_1 = df_cluster_1.explode("history_cuis")
df_referral_stats_cluster_1 = df_exploded_cluster_1["history_cuis"].value_counts().iloc[:NR_BINS_HIST]
df_hist_lookup_cluster_1 = pd.merge(df_referral_stats_cluster_1, df_lookup_data, how="left", left_index=True,
                                    right_on="CUI")
df_hist_lookup_cluster_1 = df_hist_lookup_cluster_1.reset_index(drop=True)
df_hist_lookup_cluster_1.rename(columns={"history_cuis": "count"}, inplace=True)

tab_referral_stats_col_1.write(df_hist_lookup_cluster_1[["CUI", "Name", "count"]])

fig_8 = go.Figure(go.Bar(x=df_referral_stats_cluster_1.index, y=df_referral_stats_cluster_1.values))
fig_8.update_layout(title="[Filter 1] CUI distribution history")

tab_referral_stats_col_2.plotly_chart(fig_8)

df_exploded_cluster_2 = df_cluster_2.explode("history_cuis")
df_referral_stats_cluster_2 = df_exploded_cluster_2["history_cuis"].value_counts().iloc[:NR_BINS_HIST]
df_hist_lookup_cluster_2 = pd.merge(df_referral_stats_cluster_2, df_lookup_data, how="left", left_index=True,
                                    right_on="CUI")
df_hist_lookup_cluster_2 = df_hist_lookup_cluster_2.reset_index(drop=True)
df_hist_lookup_cluster_2.rename(columns={"history_cuis": "count"}, inplace=True)

tab_referral_stats_col_1.write(df_hist_lookup_cluster_2[["CUI", "Name", "count"]])

fig_9 = go.Figure(go.Bar(x=df_referral_stats_cluster_2.index, y=df_referral_stats_cluster_2.values))
fig_9.update_layout(title="[Filter 2] CUI distribution history")

tab_referral_stats_col_2.plotly_chart(fig_9)
tab_referral_stats_col_1.write("---")
# ------------------------------------------------------------------------------------------------------------

# *****************************************
tab_output_stats.text('[Filter_1] dataframe')
tab_output_stats.write(df_cluster_1)

tab_output_stats.text('[Filter_2] dataframe')
tab_output_stats.write(df_cluster_2)
# *****************************************
# ------------------------------------------------------------------------------------------------------------

tab_tests.title("Hypothesis test statistics")
print("Calculate chi2 stats...")
df_test_stats_binary = HypothesisTestHelper.pathologies_chi2_test_binary(df_cluster_1, df_cluster_2,
                                                                         observation_classes)
df_test_stats = HypothesisTestHelper.pathologies_chi2_test(df_cluster_1, df_cluster_2, observation_classes)
print("Calculate chi2 stats done :)")
tab_tests.text("Chi2 test stats")
tab_tests.write(df_test_stats)
tab_tests.text("Chi2 test stats binary setting")
tab_tests.write(df_test_stats_binary)

# ------------------------------------------------------------------------------------------------------------
image = Image.open("./data/causal_graph.png")
tab_causal_graph.image(image, caption="Causal graph", width=1000)
# ------------------------------------------------------------------------------------------------------------
