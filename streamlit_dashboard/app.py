import configparser
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from sqlite3 import Connection
from tqdm import tqdm

tqdm.pandas()

print("Start")
config = configparser.ConfigParser()
config.read('../data/config/local_config.ini')

config_filter = configparser.ConfigParser()
config_filter.read('../data/config/streamlit_constants.ini')


tmp_ids = [
    10000032,
    10000764,
    10000898,
    10000935,
    10000980,
    10001038,
    10001122,
    10001176,
    10001217,
    10001401
]


@st.cache(allow_output_mutation=True)
def get_connection(path):
    """Put the connection in cache to reuse if path does not change."""
    print("db path: ", path)
    return sqlite3.connect(path, check_same_thread=False)


@st.cache(hash_funcs={Connection: id})
def load_data(engine):
    print("load_data()")
    df_data = pd.read_sql('SELECT * FROM "mimic-cxr-2.0.0-chexpert"', con=engine)
    return df_data


@st.cache(hash_funcs={Connection: id})
def load_timeline_data(engine):
    print("load_timeline_data()")
    df_timeline_data = pd.read_sql('select * from streamlit_timeline_data', con=engine)
    df_timeline_data['datetime_start'] = pd.to_datetime(df_timeline_data['datetime_start'], infer_datetime_format=True)
    df_timeline_data['datetime_end'] = pd.to_datetime(df_timeline_data['datetime_end'], infer_datetime_format=True)
    df_timeline_data['datetime_start_norm'] = pd.to_datetime(df_timeline_data['datetime_start_norm'],
                                                             infer_datetime_format=True)
    df_timeline_data['datetime_end_norm'] = pd.to_datetime(df_timeline_data['datetime_end_norm'],
                                                           infer_datetime_format=True)

    df_stay_id_timeline_id = pd.read_sql('select study_id, timeline_id from streamlit_timeline_data', con=conn)
    df_report_label = pd.read_sql('select * from "mimic-cxr-2.0.0-chexpert"', con=conn)

    return df_timeline_data, df_stay_id_timeline_id, df_report_label


@st.cache
def calc_accommodation_stats_data(df, df_stay_id_timeline_id, df_report_label):
    df_tmp = df.groupby('subject_id')['timeline_id'].agg(set)
    df_tmp = pd.DataFrame(df_tmp).reset_index()
    flattened_col = pd.DataFrame(
        [(index, value) for (index, values) in df_tmp['timeline_id'].iteritems() for value in values],
        columns=['index', 'timeline_id']).set_index('index')
    df_output = df_tmp.drop('timeline_id', axis=1).join(flattened_col)
    df_output = pd.merge(df_output, df_stay_id_timeline_id, how='left', on="timeline_id").drop_duplicates()
    df_output = pd.merge(df_output, df_report_label, how='left', on="study_id").drop_duplicates()
    df_output = df_output.rename({'subject_id_x': 'subject_id'}, axis=1)
    df_output.drop('subject_id_y', axis=1, inplace=True)

    def _apply_process_type(row):
        if row['timeline_id'].startswith('h_'):
            row['is_hospital'] = True
            row['is_ed'] = False
            row['is_xray'] = False
        elif row['timeline_id'].startswith('e_'):
            row['is_hospital'] = False
            row['is_ed'] = True
            row['is_xray'] = False
        elif row['timeline_id'].startswith('x_'):
            row['is_hospital'] = False
            row['is_ed'] = False
            row['is_xray'] = True
        return row

    df_output = df_output.progress_apply(_apply_process_type, axis=1)

    return df_output


conn = get_connection(config['DATABASE']['path'])

finding_columns = ['Atelectasis', 'Cardiomegaly',
                   'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum', 'Fracture',
                   'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                   'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']

st.title('MIMIC-CXR Dashboard')

options1 = st.sidebar.multiselect(
    'Filter by columns',
    ['patient', 'datetime_start', 'datetime_end', 'process_type', 'findings'],
    ['findings'])

text_filter_study_id_str = ''
if 'study_id_str' in options1:
    text_filter_study_id_str = st.sidebar.text_input('Study ID filter', '')

text_filter_subject_id = ''
if 'patient' in options1:
    text_filter_subject_id = st.sidebar.text_input('Subject ID filter', '')

findings_filter = []
findings_filter_mode = []
if 'findings' in options1:
    list_filter = json.loads(config_filter['FINDINGS']['filter'])
    findings_filter = st.sidebar.multiselect(
        'Filter by columns',
        list_filter,
        ['PRESENT_Atelectasis'])
    findings_filter_mode = st.sidebar.selectbox(
        'AND/OR',
        ('AND', 'OR'))

print(options1)

trigger_filter = st.sidebar.checkbox("Trigger timeline plot")

if True:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    df_data = load_data(conn)
    df_timeline_data, df_stay_id_timeline_id, df_report_label = load_timeline_data(conn)
    df_accomm_stats = calc_accommodation_stats_data(df_timeline_data, df_stay_id_timeline_id, df_report_label)
    data_load_state.text('Loading data...done!')
    print('Done')

    df_tmp = df_timeline_data.copy(deep=True)
    print(df_tmp.shape)
    if len(text_filter_study_id_str) != 0:
        print('filter study_id')
        df_tmp = df_tmp[df_tmp['study_id_str'] == "patient_" + text_filter_study_id_str]
    if len(text_filter_subject_id) != 0:
        print('filter subject_id')
        df_tmp = df_tmp[df_tmp['subject_id'] == int(text_filter_subject_id)]
        df_data = df_data[df_data['subject_id'] == int(text_filter_subject_id)]
        df_accomm_stats = df_accomm_stats[df_accomm_stats['subject_id'] == int(text_filter_subject_id)]
    print(text_filter_subject_id)
    print(df_tmp.shape)

    if len(findings_filter) > 0:
        for e in findings_filter:
            if e.startswith('PRESENT'):
                df_tmp = df_tmp[df_tmp[e[8:]] == 1]
                df_accomm_stats = df_accomm_stats[df_accomm_stats[e[8:]] == 1]
            if e.startswith('NO'):
                df_tmp = df_tmp[df_tmp[e[3:]] == 0]
                df_accomm_stats = df_accomm_stats[df_accomm_stats[e[3:]] == 0]
            if e.startswith('UNCERTAIN'):
                df_tmp = df_tmp[df_tmp[e[10:]] == -1]
                df_accomm_stats = df_accomm_stats[df_accomm_stats[e[10:]] == -1]

    dict_general_stats = {
        'examinations': ['between 2011-2016'],
        'distinct patients': [df_data['subject_id'].nunique()],
        'distinct studies': [df_data['study_id'].nunique()],
    }

    print(dict_general_stats)
    st.table(pd.DataFrame(dict_general_stats).T)

    # st.table(df_data.head(10))
    # st.table(df_timeline_data.head(10))


    df_hist = df_accomm_stats[finding_columns].apply(pd.Series.value_counts)

    fig = go.Figure(data=[go.Bar(name=e, x=df_hist.index.values, y=df_hist[e].values) for e in finding_columns])
    fig.update_layout(barmode='group')
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 0, -1],
            ticktext=['is present', 'not present', 'uncertain']
        )
    )
    st.plotly_chart(fig)

    df_accomm_stats_hist = df_accomm_stats.groupby('subject_id')[['is_hospital', 'is_ed', 'is_xray']].sum().reset_index()
    df_hist_accomm = df_accomm_stats_hist[['is_hospital', 'is_ed', 'is_xray']].sum()
    print(df_hist_accomm)

    fig_a = go.Figure([go.Bar(x=df_hist_accomm.index.values, y=df_hist_accomm.values)])
    st.plotly_chart(fig_a)

    fig12 = px.histogram(df_tmp[df_tmp['process_type'] != 'xray'],
                         x='stay_duration',
                         color='process_type',
                         nbins=2500,
                         range_x=[0, 3e3])
    st.plotly_chart(fig12)

    stats_dict = {
        'mean_ed': [],
        'median_ed': [],
        'variance_ed': [],
        'mean_hosp': [],
        'median_hosp': [],
        'variance_hosp': [],
        'xray_samples': [],
        'ed_samples': [],
        'hosp_samples': [],
        'total_samples': [],
    }

    stats_dict['mean_ed'].append(str("%.2f" % df_tmp[df_tmp['process_type'] == 'edstay']['stay_duration'].mean()))
    stats_dict['median_ed'].append(str("%.2f" % df_tmp[df_tmp['process_type'] == 'edstay']['stay_duration'].median()))
    stats_dict['variance_ed'].append(str("%.2f" % df_tmp[df_tmp['process_type'] == 'edstay']['stay_duration'].var()))
    stats_dict['mean_hosp'].append(str("%.2f" % df_tmp[df_tmp['process_type'] == 'admission']['stay_duration'].mean()))
    stats_dict['median_hosp'].append(
        str("%.2f" % df_tmp[df_tmp['process_type'] == 'admission']['stay_duration'].median()))
    stats_dict['variance_hosp'].append(
        str("%.2f" % df_tmp[df_tmp['process_type'] == 'admission']['stay_duration'].var()))

    stats_dict['xray_samples'].append(df_tmp[df_tmp['process_type'] == 'xray'].shape[0])
    stats_dict['ed_samples'].append(df_tmp[df_tmp['process_type'] == 'edstay'].shape[0])
    stats_dict['hosp_samples'].append(df_tmp[df_tmp['process_type'] == 'admission'].shape[0])
    stats_dict['total_samples'].append(df_tmp.shape[0])

    st.table(pd.DataFrame(stats_dict).astype(str).T)

    if ('study_id_str' not in options1) and ('patient' not in options1) and ('findings' not in options1):
        df_tmp = df_tmp[df_tmp['subject_id'].isin(tmp_ids)]  # filter by specific ids because of visualization runtime

    if trigger_filter:
        fig1 = px.timeline(df_tmp, x_start='datetime_start_norm', x_end='datetime_end_norm', y='timeline_id',
                           hover_name='subject_id', color='process_type', opacity=0.5, width=1000, height=750)
        st.plotly_chart(fig1)

# *****************************************************************************************************************
# *****************************************************************************************************************
if False:
    add_selectbox = st.sidebar.selectbox(
        "x-axis",
        df_data.columns.values,
        3
    )

    add_color_select = st.sidebar.selectbox(
        "color",
        np.concatenate((np.array(['None']), df_data.columns.values)),
        0
    )

    options1 = st.sidebar.multiselect(
        'Select columns to filter',
        ['KURZANAMNESE', 'BEFUND_TEXT', 'FRAGESTELLUNG'],
        ['KURZANAMNESE'])

    text_filter_study_id_str = st.sidebar.text_input('Text filter', '')

    st.subheader('Histogram')

    if len(text_filter_study_id_str) > 0 and len(options1) > 0:
        df_data['filter_text'] = df_data[options1].values.tolist()
        df_data['filter_text'] = df_data['filter_text'].apply(lambda e: ' '.join(e))
        df_data = df_data[
            df_data['filter_text'].str.contains(text_filter_study_id_str.lower(), case=False, na=False, regex=True)]

    if add_color_select != 'None':
        df_gr2 = df_data.groupby(by=[add_color_select, add_selectbox]).size()
    else:
        df_gr2 = df_data.groupby(by=[add_selectbox]).size()

    df_gr2 = df_gr2.reset_index().rename(columns={0: 'count'})

    fig = px.bar(df_hist,
                 x=df_hist.index.values,
                 y=df_hist['Atelectasis'].values)

    st.plotly_chart(fig)

    st.write('number of df_gr2  samples: ', df_gr2.shape[0])
    st.write('number of df_data samples: ', df_data.shape[0])

    if st.checkbox('Show filtered dataframe'):
        st.write(df_data)

    if len(text_filter_study_id_str) > 0:
        st.write('filtered by KURZANAMNESE: ', text_filter_study_id_str)
