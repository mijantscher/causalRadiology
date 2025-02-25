import pandas as pd
import numpy as np
import configparser
import sqlite3
from tqdm import tqdm

tqdm.pandas()


def timeline_id(dict_regroup_timeline, df):
    df_sub = df
    dict_regroup_timeline["subject_id"] += df_sub["prim_subject_id"].values.tolist()
    dict_regroup_timeline["study_id"] += df_sub["prim_study_id"].values.tolist()
    dict_regroup_timeline["stay_id"] += df_sub["stay_id"].values.tolist()
    dict_regroup_timeline["hadm_id"] += df_sub["hadm_id"].values.tolist()
    dict_regroup_timeline["process_id"] += df_sub["prim_study_id"].values.tolist()
    dict_regroup_timeline["timeline_id"] += df_sub["timeline_id"].values.tolist()
    dict_regroup_timeline["datetime_start"] += df_sub["study_date"].values.tolist()
    dict_regroup_timeline["datetime_end"] += df_sub["study_date_end"].values.tolist()
    dict_regroup_timeline["process_type"] += ['xray'] * df_sub.shape[0]
    dict_regroup_timeline["Atelectasis"] += df_sub["Atelectasis"].values.tolist()
    dict_regroup_timeline["Cardiomegaly"] += df_sub["Cardiomegaly"].values.tolist()
    dict_regroup_timeline["Consolidation"] += df_sub["Consolidation"].values.tolist()
    dict_regroup_timeline["Edema"] += df_sub["Edema"].values.tolist()
    dict_regroup_timeline["Enlarged_Cardiomediastinum"] += df_sub[
        "Enlarged_Cardiomediastinum"].values.tolist()
    dict_regroup_timeline["Fracture"] += df_sub["Fracture"].values.tolist()
    dict_regroup_timeline["Lung_Lesion"] += df_sub["Lung_Lesion"].values.tolist()
    dict_regroup_timeline["Lung_Opacity"] += df_sub["Lung_Opacity"].values.tolist()
    dict_regroup_timeline["No_Finding"] += df_sub["No_Finding"].values.tolist()
    dict_regroup_timeline["Pleural_Effusion"] += df_sub["Pleural_Effusion"].values.tolist()
    dict_regroup_timeline["Pleural_Other"] += df_sub["Pleural_Other"].values.tolist()
    dict_regroup_timeline["Pneumonia"] += df_sub["Pneumonia"].values.tolist()
    dict_regroup_timeline["Pneumothorax"] += df_sub["Pneumothorax"].values.tolist()
    dict_regroup_timeline["Support_Devices"] += df_sub["Support_Devices"].values.tolist()

    df_sub = df[df['stay_id'] > 0]
    dict_regroup_timeline["subject_id"] += df_sub["prim_subject_id"].values.tolist()
    dict_regroup_timeline["study_id"] += df_sub["prim_study_id"].values.tolist()
    dict_regroup_timeline["stay_id"] += df_sub["stay_id"].values.tolist()
    dict_regroup_timeline["hadm_id"] += df_sub["hadm_id"].values.tolist()
    dict_regroup_timeline["process_id"] += df_sub["stay_id"].values.tolist()
    dict_regroup_timeline["timeline_id"] += df_sub["timeline_id"].values.tolist()
    dict_regroup_timeline["datetime_start"] += df_sub["intime"].values.tolist()
    dict_regroup_timeline["datetime_end"] += df_sub["outtime"].values.tolist()
    dict_regroup_timeline["process_type"] += ['edstay'] * df_sub.shape[0]
    dict_regroup_timeline["Atelectasis"] += df_sub["Atelectasis"].values.tolist()
    dict_regroup_timeline["Cardiomegaly"] += df_sub["Cardiomegaly"].values.tolist()
    dict_regroup_timeline["Consolidation"] += df_sub["Consolidation"].values.tolist()
    dict_regroup_timeline["Edema"] += df_sub["Edema"].values.tolist()
    dict_regroup_timeline["Enlarged_Cardiomediastinum"] += df_sub[
        "Enlarged_Cardiomediastinum"].values.tolist()
    dict_regroup_timeline["Fracture"] += df_sub["Fracture"].values.tolist()
    dict_regroup_timeline["Lung_Lesion"] += df_sub["Lung_Lesion"].values.tolist()
    dict_regroup_timeline["Lung_Opacity"] += df_sub["Lung_Opacity"].values.tolist()
    dict_regroup_timeline["No_Finding"] += df_sub["No_Finding"].values.tolist()
    dict_regroup_timeline["Pleural_Effusion"] += df_sub["Pleural_Effusion"].values.tolist()
    dict_regroup_timeline["Pleural_Other"] += df_sub["Pleural_Other"].values.tolist()
    dict_regroup_timeline["Pneumonia"] += df_sub["Pneumonia"].values.tolist()
    dict_regroup_timeline["Pneumothorax"] += df_sub["Pneumothorax"].values.tolist()
    dict_regroup_timeline["Support_Devices"] += df_sub["Support_Devices"].values.tolist()

    df_sub = df[df['hadm_id'] > 0]
    dict_regroup_timeline["subject_id"] += df_sub["prim_subject_id"].values.tolist()
    dict_regroup_timeline["study_id"] += df_sub["prim_study_id"].values.tolist()
    dict_regroup_timeline["stay_id"] += df_sub["stay_id"].values.tolist()
    dict_regroup_timeline["hadm_id"] += df_sub["hadm_id"].values.tolist()
    dict_regroup_timeline["process_id"] += df_sub["hadm_id"].values.tolist()
    dict_regroup_timeline["timeline_id"] += df_sub["timeline_id"].values.tolist()
    dict_regroup_timeline["datetime_start"] += df_sub["admittime"].values.tolist()
    dict_regroup_timeline["datetime_end"] += df_sub["dischtime"].values.tolist()
    dict_regroup_timeline["process_type"] += ['admission'] * df_sub.shape[0]
    dict_regroup_timeline["Atelectasis"] += df_sub["Atelectasis"].values.tolist()
    dict_regroup_timeline["Cardiomegaly"] += df_sub["Cardiomegaly"].values.tolist()
    dict_regroup_timeline["Consolidation"] += df_sub["Consolidation"].values.tolist()
    dict_regroup_timeline["Edema"] += df_sub["Edema"].values.tolist()
    dict_regroup_timeline["Enlarged_Cardiomediastinum"] += df_sub[
        "Enlarged_Cardiomediastinum"].values.tolist()
    dict_regroup_timeline["Fracture"] += df_sub["Fracture"].values.tolist()
    dict_regroup_timeline["Lung_Lesion"] += df_sub["Lung_Lesion"].values.tolist()
    dict_regroup_timeline["Lung_Opacity"] += df_sub["Lung_Opacity"].values.tolist()
    dict_regroup_timeline["No_Finding"] += df_sub["No_Finding"].values.tolist()
    dict_regroup_timeline["Pleural_Effusion"] += df_sub["Pleural_Effusion"].values.tolist()
    dict_regroup_timeline["Pleural_Other"] += df_sub["Pleural_Other"].values.tolist()
    dict_regroup_timeline["Pneumonia"] += df_sub["Pneumonia"].values.tolist()
    dict_regroup_timeline["Pneumothorax"] += df_sub["Pneumothorax"].values.tolist()
    dict_regroup_timeline["Support_Devices"] += df_sub["Support_Devices"].values.tolist()


print("Start")
config = configparser.ConfigParser()
config.read('../../data/config/local_config.ini')
conn = sqlite3.connect(config['DATABASE']['path'], check_same_thread=False)

df_timeline_data = pd.read_sql(
    'select t1.subject_id as prim_subject_id, t1.study_id as prim_study_id, t1.study_date, t1.stay_id, t1.hadm_id, e.intime, e.outtime, a.admittime, a.dischtime, findings.* from tmp_timeline as t1 left join edstays e on t1.stay_id = e.stay_id left join admissions a on t1.hadm_id = a.hadm_id left join "mimic-cxr-2.0.0-chexpert" findings on t1.study_id = findings.study_id',
    con=conn)
df_timeline_data['study_date'] = pd.to_datetime(df_timeline_data['study_date'], infer_datetime_format=True)
df_timeline_data['intime'] = pd.to_datetime(df_timeline_data['intime'], infer_datetime_format=True)
df_timeline_data['outtime'] = pd.to_datetime(df_timeline_data['outtime'], infer_datetime_format=True)
df_timeline_data['admittime'] = pd.to_datetime(df_timeline_data['admittime'], infer_datetime_format=True)
df_timeline_data['dischtime'] = pd.to_datetime(df_timeline_data['dischtime'], infer_datetime_format=True)

dict_regroup_timeline = {
    "subject_id": [],
    "study_id": [],
    "stay_id": [],
    "hadm_id": [],
    "process_id": [],
    "timeline_id": [],
    "datetime_start": [],
    "datetime_end": [],
    "process_type": [],
    "Atelectasis": [],
    "Cardiomegaly": [],
    "Consolidation": [],
    "Edema": [],
    "Enlarged_Cardiomediastinum": [],
    "Fracture": [],
    "Lung_Lesion": [],
    "Lung_Opacity": [],
    "No_Finding": [],
    "Pleural_Effusion": [],
    "Pleural_Other": [],
    "Pneumonia": [],
    "Pneumothorax": [],
    "Support_Devices": [],
}

df_timeline_data['study_timedelta'] = pd.to_timedelta('1h')
df_timeline_data['study_date_end'] = df_timeline_data['study_date'] + df_timeline_data['study_timedelta']


def _apply_timeline_id(row):
    if row['hadm_id'] > 0:
        return 'h_' + str(row['hadm_id'])
    elif (row['stay_id'] > 0) and (row['hadm_id'] <= 0):
        return 'e_' + str(row['stay_id'])
    else:
        return 'x_' + str(row['prim_study_id'])


df_timeline_data['timeline_id'] = df_timeline_data.progress_apply(_apply_timeline_id, axis=1)
timeline_id(dict_regroup_timeline, df_timeline_data)
df_timeline_data = pd.DataFrame.from_dict(dict_regroup_timeline)

df_timeline_data['datetime_start'] = pd.to_datetime(df_timeline_data['datetime_start'], infer_datetime_format=True)
df_timeline_data['datetime_end'] = pd.to_datetime(df_timeline_data['datetime_end'], infer_datetime_format=True)
df_timeline_data['stay_duration'] = (df_timeline_data['datetime_end'] - df_timeline_data[
    'datetime_start']) / np.timedelta64(1, 'h')

df_timeline_data.drop_duplicates(['timeline_id', 'datetime_start', 'datetime_end', 'process_type'], inplace=True)

dt = pd.Timestamp('2020-01-01')


def _transform_datetime(df_group):
    time_delta = df_group.min() - dt
    return time_delta


df_timeline_data['time_delta'] = df_timeline_data.groupby('timeline_id')['datetime_start'].progress_transform(
    _transform_datetime)
df_timeline_data['datetime_start_norm'] = df_timeline_data['datetime_start'] - df_timeline_data['time_delta']
df_timeline_data['datetime_end_norm'] = df_timeline_data['datetime_end'] - df_timeline_data['time_delta']

df_timeline_data.to_sql('streamlit_timeline_data', con=conn, if_exists='replace', index_label=False)
print('Done')
