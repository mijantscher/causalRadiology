import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read('../../data/config/local_config.ini')

conn = sqlite3.connect(config['DATABASE']['path'])
tqdm.pandas()

df_cxr_metadata = pd.read_sql(
    'select subject_id, study_id, StudyDate, StudyTime, count(study_id) as nr_studies from "mimic-cxr-2.0.0-metadata" group by study_id order by subject_id asc',
    con=conn)
df_cxr_metadata['StudyTime1'] = df_cxr_metadata['StudyTime'].astype(int).astype(str).apply(lambda e: e.zfill(6))
df_cxr_metadata['StudyDateTime'] = df_cxr_metadata['StudyDate'].astype(str) + ":" + df_cxr_metadata['StudyTime1']
df_cxr_metadata['study_date'] = pd.to_datetime(df_cxr_metadata['StudyDateTime'], format="%Y%m%d:%H%M%S")
i = 0

df_edstays = pd.read_sql(
    f'select * from edstays where subject_id in ({", ".join(map(str, df_cxr_metadata["subject_id"].unique()))})',
    con=conn)
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'], infer_datetime_format=True)
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'], infer_datetime_format=True)

df_admissions = pd.read_sql(
    f'select * from admissions where subject_id in ({", ".join(map(str, df_cxr_metadata["subject_id"].unique()))})',
    con=conn)
df_admissions['admittime'] = pd.to_datetime(df_admissions['admittime'], infer_datetime_format=True)
df_admissions['dischtime'] = pd.to_datetime(df_admissions['dischtime'], infer_datetime_format=True)


# now we try to get the hadm_id. two possible scenarios:
#   from table edstays when intime < study_date < outtime
#   from table admissions when admittime < study_date < dischtime


def _apply_hadm_id(row):
    stay_id = -1

    df_pat_id = df_edstays[(df_edstays['subject_id'] == row['subject_id']) &
                           (df_edstays['intime'] < row['study_date']) &
                           (df_edstays['outtime'] > row['study_date'])]

    if df_pat_id.shape[0] == 1:
        stay_id = df_pat_id.iloc[0]['stay_id']

    # if xray was not taken in the ED, it must have been taken during the hospital stay (admissions)
    if df_pat_id.shape[0] == 0:
        df_pat_id = df_admissions[(df_admissions['subject_id'] == row['subject_id']) &
                                  (df_admissions['admittime'] < row['study_date']) &
                                  (df_admissions['dischtime'] > row['study_date'])]
    if df_pat_id.shape[0] == 0:
        # print('Neither ED nor Admission requested a xray', row)
        row['stay_id'] = stay_id
        row['hadm_id'] = -1
        return row
        # TODO instead have a look in the ICU table if the imaging observation is done during this stay?
    if df_pat_id.shape[0] != 1:
        print('Multiple entries for hadm_id', row)

    hadm_id = int(df_pat_id.iloc[0]['hadm_id']) if not np.isnan(df_pat_id.iloc[0][
                                                                    'hadm_id']) else -2  # if patient was only admitted to the ED but not to the hospital afterwards
    row['stay_id'] = stay_id
    row['hadm_id'] = hadm_id
    return row


# df_cxr_metadata['hadm_id'] = df_cxr_metadata.apply(_apply_hadm_id, axis=1)
# df_cxr_metadata = df_cxr_metadata.head(2000)
# df_cxr_metadata['stay_id'], df_cxr_metadata['hadm_id'] = df_cxr_metadata.progress_apply(_apply_hadm_id, axis=1)
df_cxr_metadata = df_cxr_metadata.progress_apply(_apply_hadm_id, axis=1)
df_cxr_metadata.to_sql('tmp_timeline', con=conn, index=False, chunksize=1000, if_exists='replace')
