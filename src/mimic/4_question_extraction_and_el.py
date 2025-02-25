import numpy as np
import pandas as pd
import sqlite3
import configparser
import datetime
import spacy


def detect_omni_cq_ids(note):
    """inputs a potentially ominous note then splits the note into [note, clinical_question, num_iden_words, iden_words] where every entrie could be None
    if an identification word is found "?" do not split the note if they appear within 0.4 * len(note) or at note[-1]
    Note 1: ignores cq markers "//", they are handled in sep_clinical_question, which uses this function
    Note 2: The function is more complicated than necessary(probably) in order to get all present iden_words
    outputs note, cq, number of present identification words, present identification words ("?" wont be added in all cases)
    """
    # iden_words_found gives a list with the index of the iden_word in the note, if the word is not present i = -1
    lower_note = note.lower()
    iden_words_found = np.array([lower_note.find(iden_word) for iden_word in identification_words])

    # after looking closly at 327 samples without "//" but with "?" in them, this seemed like the most reasonable approach do deal with "?":
    if iden_words_found[0] != -1:  # if a "?" is present
        ignore_till_i = int(0.4 * len(note))  # not all "?" are proper cq indicators

        if iden_words_found[0] < ignore_till_i:
            iden_words_found[0] = note.find("?", ignore_till_i)  # often times there are several "?" in 1 note

        if iden_words_found[0] == len(note) - 1:  # dont split for an "?" at the end of the note
            iden_words_found[0] = -1

    iden_words_found_mask = iden_words_found != -1
    i_iden_word = iden_words_found[iden_words_found_mask]
    if i_iden_word.size == 0:  # do this case first because it will probably occour more often
        out_note, out_cq, iden_words = note, None, None

    else:
        i_iden_word = i_iden_word.min()  # the fist iden_word in the note is used
        out_note = note[:i_iden_word]
        out_cq = note[i_iden_word:]
        iden_words = np.array(identification_words)[iden_words_found_mask]

    num_iden_words = iden_words_found_mask.sum()
    return out_note, out_cq, num_iden_words, iden_words


def history_detect(note):
    """inputs note and splits for "History:"
    outputs [note, history] where note and/or history can be None  """
    i_hist = note.find("History:")
    if i_hist == -1:
        return note, None
    else:
        out_note, out_hist = note[:i_hist], note[i_hist:]
        if len(out_note) == 0:
            out_note = None
        return out_note, out_hist


def sep_clinical_question(note):
    """takes in a note and checks: for "//" (good clinical question (cq) format), for bad cq format and also for "history:"
    then outputs [Note, cq, history, num_iden_words, iden_words] where every entrie could be None """
    if note:
        output = note.split(sep="//", maxsplit=1)  # -1 means all splits, idealy "//" should occur once per note at max

        num_iden_words, iden_words = None, None
        if len(output) == 1:  # if no good cq format present check for omni format
            out_note, out_cq, num_iden_words, iden_words = detect_omni_cq_ids(output[0])
        elif len(output) == 2:
            out_note, out_cq = output[0], output[1]
        else:
            error_text = f"There is a note where the marker // apears 2 times. This case is not covered.\nNote: {note}"
            raise ValueError(error_text)

        out_note, out_hist = history_detect(out_note)
        return out_note, out_cq, out_hist, num_iden_words, iden_words  # iden_words are only shown for bad cq format
    else:
        return [None] * 5


def get_cuis_per_col(series_pd):
    """
    changed this to recieve a pd.series of notes instead of 1 note to use the
    nlp.pipe() feature which makes everything faster

    takes a note and performs entity linking on it, then outputs a list of tuples
    [(CUI, enitity), ...] with the CUI nummer with the highest score for each entity
    if the note is none output [['no_note', 'no_note']]
    if there is no entity detect in the note it outputs [['no_entities_detected', 'no_entities_detected']]
    if one enity is unkown to the Linker a ["entity_unknown", "entity_unknown"] is placed instead of CUI number and entity"""
    series_pd = series_pd.to_list()
    series_pd = [note if note else "" for note in series_pd]

    form = lambda ent: ent.text if ent.text else None  # highlits/prevents artefacts not sure if still necessary

    new_column = []
    for doc in nlp.pipe(series_pd):
        ents_doc = doc.ents
        if not doc.text:
            new_column.append([["no_note"] * 2])
        elif ents_doc:  # do_ent._.kb_ents contains all (cui, probability) pairs of 1 entity
            cui_ent = [(do_ent._.kb_ents[0][0], form(do_ent)) if do_ent._.kb_ents else ["entity_unknown", form(do_ent)]
                       for do_ent in ents_doc]
            new_column.append(cui_ent)
        else:
            new_column.append([["no_entities_detected"] * 2])

    return new_column


def get_cuis_old(note):
    """takes a note and performs entity linking on it, then outputs a list of tuples
    [(CUI, enitity), ...] with the CUI nummer with the highest score for each entity
    if the note is none output [['no_note', 'no_note']]
    if there is no entity detect in the note it outputs [['no_entities_detected', 'no_entities_detected']]
    if one enity is unkown to the Linker a ["entity_unknown", "entity_unknown"] is placed instead of CUI number and entity"""
    if not note:
        return [["no_note"] * 2]

    doc = nlp(note)

    ents_doc = doc.ents
    if ents_doc:
        return [(do_ent._.kb_ents[0][0], do_ent) if do_ent._.kb_ents else ["entity_unknown", do_ent] for do_ent in
                ents_doc]
    else:
        return [["no_entities_detected"] * 2]


def combine_str_columns(left, right):
    """outputs a potentially combined string or None"""
    try:
        match bool(left), bool(right):
            case (0, 0):
                out = None
            case (1, 0):
                out = left
            case (0, 1):
                out = right
            case (1, 1):
                out = left + " " + right
    except:
        if left and not right:
            out = left
        if right and not left:
            out = right
        if not left and not right:
            out = None
        if left and right:
            out = left + " merged " + right
    return out


def create_goal_df_no_cuis(df):
    """takes in raw data eg.: df_sectioned_reports then seperates the data in the
    indication column and creates some new columns for the seperated data
    returns df"""
    series_of_tupl_list = df.indication.apply(sep_clinical_question)

    get_column = lambda c: [stuff[c] for stuff in series_of_tupl_list]
    goal_df = df[["study", "history", "comparison"]]
    goal_df["indic_filtered"] = get_column(0)
    goal_df["clinical_question"] = get_column(1)
    goal_df["hist_from_ind"] = get_column(2)
    # goal_df["num_ident_words"] = get_column(3)
    # goal_df["ident_words"] = get_column(4)

    zip_hists = zip(goal_df.hist_from_ind, goal_df.history)
    goal_df["hist_finished"] = [combine_str_columns(left, right) for left, right in zip_hists]

    return goal_df[["study", "indic_filtered", "clinical_question", "hist_finished", "comparison"]]


def create_goal_df_cuis(df):
    """takes in a df and adds columns with CUI numbers"""
    started_experiment = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    print(
        f"Started using the EntityLinker on the data at: {started_experiment.replace(microsecond=0).replace(second=0)}")

    goal_df = df  # .copy()
    goal_df["indication_cuis_ents"] = get_cuis_per_col(df.indic_filtered)
    print("\n - finished cuis on indication -\n")
    goal_df["clinical_q_cuis_ents"] = get_cuis_per_col(df.clinical_question)
    print("\n - finished cuis on clinical_question -\n")
    goal_df["history_cuis_ents"] = get_cuis_per_col(df.hist_finished)
    print("\n - finished cuis on history -\n")
    goal_df["comparison_cuis_ents"] = get_cuis_per_col(df.comparison)
    print("\n - finished cuis on comparison -\n")

    columns = ["study", "indic_filtered", "indication_cuis_ents", "clinical_question", "clinical_q_cuis_ents",
               "hist_finished", "history_cuis_ents", "comparison", "comparison_cuis_ents"]

    ended_exp = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    print(f"\nFinished to process the data at: {ended_exp.replace(microsecond=0).replace(second=0)}")
    return goal_df[columns]


def get_column(c, df_series):
    column = []
    for stuff in df_series:
        if stuff:
            new_row = []
            for pair in stuff:
                new_row.append(pair[c])
            column.append(new_row)
        else:
            column.append(None)
    return column


def edit_goal_df_cuis(df):
    """separates the ..._cuis_ents columns such that e.g.: 1 row of indication_cuis_ents
    would become indication_cuis: [cui_1, cui_2,..] and indication_ents: [ent_1, ent_2, ..]"""

    df["indication_cuis"] = get_column(0, df.indication_cuis_ents)
    df["indication_ents"] = get_column(1, df.indication_cuis_ents)

    df["clinical_q_cuis"] = get_column(0, df.clinical_q_cuis_ents)
    df["clinical_q_ents"] = get_column(1, df.clinical_q_cuis_ents)

    df["history_cuis"] = get_column(0, df.history_cuis_ents)
    df["history_ents"] = get_column(1, df.history_cuis_ents)

    df["comparison_cuis"] = get_column(0, df.comparison_cuis_ents)
    df["comparison_ents"] = get_column(1, df.comparison_cuis_ents)

    columns = ["study", "indic_filtered", "indication_cuis", "indication_ents",
               "clinical_question", "clinical_q_cuis", "clinical_q_ents",
               "hist_finished", "history_cuis", "history_ents", "comparison", "comparison_cuis", "comparison_ents"]

    return df[columns]


def combine_and_save(txt_or_json):
    """inputs "txt" or "json" then loads the parts of the goal_df from the current directory,
    concatenates them and saves them as on dataframe
    also returns the goal_df
    """
    if txt_or_json not in ["txt", "json"]:
        raise ValueError(f"{txt_or_json} is not supported. Use txt or json.")

    how_many_parts_finished = 0
    section_is = [25000 * i for i in range(how_many_parts_finished, 10)]
    section_is += [
        len(df_sectioned_reports)]  # section_is equals [0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 227614]

    goal_df_names = [f"goal_df_i_{section_is[n]}_untill_{section_is[n + 1]}.{txt_or_json}" for n in
                     range(len(section_is) - 1)]

    if txt_or_json == "txt":
        complete_goal_df_cuis = [pd.read_csv(name, sep=str("🎃"), engine="python") for name in goal_df_names]
        complete_goal_df_cuis = pd.concat(complete_goal_df_cuis)
        complete_goal_df_cuis = complete_goal_df_cuis.rename(columns={"Unnamed: 0": "index"}).set_index("index")

        complete_goal_df_cuis.to_csv("complete_goal_df_cuis_fromCLEAN_V4_3" + ".txt", sep=str("🎃"))

    elif txt_or_json == "json":
        complete_goal_df_cuis = [pd.read_json(name) for name in goal_df_names]
        complete_goal_df_cuis = pd.concat(complete_goal_df_cuis)

        complete_goal_df_cuis.to_json("complete_goal_df_cuis_withEmptyStrings_fromCLEAN_V4_3" + ".json",
                                      default_handler=str)

    return complete_goal_df_cuis


def create_lookup_set(path_to_df):
    df = pd.read_csv(path_to_df, sep='🎃', engine="python")
    df = df.set_index("index")

    all_used_cuis = []
    columns = [df.indication_cuis, df.clinical_q_cuis, df.history_cuis, df.comparison_cuis]
    for column in columns:
        for mini_cui_list in column:
            if mini_cui_list:
                mini_cui_list = eval(mini_cui_list)
                all_used_cuis.extend(mini_cui_list)
    return set(all_used_cuis)


def create_lookup_table(some_cui_set):
    some_cui_set = list(some_cui_set)
    for no_cui in ["no_entities_detected", "no_note", "entity_unknown"]:
        some_cui_set.remove(no_cui)

    name_list = [linker.kb.cui_to_entity[cui].canonical_name for cui in some_cui_set]

    final_table = pd.DataFrame({"CUI": some_cui_set, "Name": name_list})
    final_table.sort_values('Name', ascending=True, inplace=True)
    final_table.to_csv("look_up_table_fromClean_V4_3.txt", sep=str("🎃"))
    final_table.to_excel("look_up_table_fromClean_V4_3.xlsx")
    return final_table.reset_index(drop=True)


def change_cuis(cuis_ser, ents_ser):
    """
    inputs a cui series with lists of cuis and a ents series with lists of ents
    searches through the ents for "F" and "M" and replaces them with the CUIs for female and male
    outputs the new cuis_list
    """
    cuis_list = cuis_ser.tolist()
    ents_list = ents_ser.tolist()

    cuis_list_new = []

    for i in range(len(cuis_list)):
        cuis_mini_list = cuis_list[i]
        inds_F = [j for j, x in enumerate(ents_list[i]) if x == 'F']
        inds_M = [j for j, x in enumerate(ents_list[i]) if x == 'M']

        range_obj = range(len(cuis_mini_list))
        if inds_F:
            cuis_mini_list = [cuis_mini_list[j] if j not in inds_F else "C0043210" for j in range_obj]
        if inds_M:
            cuis_mini_list = [cuis_mini_list[j] if j not in inds_M else "C0086582" for j in range_obj]

        cuis_list_new.append(cuis_mini_list)

    return cuis_list_new


def apply_change_cuis(some_df):
    """inputs a goal_df applies change_cuis() for every CUI column
      outputs goal_df with changed cuis"""

    columns_wo_cuis = ["study", "indic_filtered", "indication_ents", "clinical_question",
                       "clinical_q_ents", "hist_finished", "history_ents", "comparison", "comparison_ents"]
    new_df = some_df[columns_wo_cuis].copy()

    new_df["indication_cuis"] = change_cuis(some_df.indication_cuis, some_df.indication_ents)
    new_df["clinical_q_cuis"] = change_cuis(some_df.clinical_q_cuis, some_df.clinical_q_ents)
    new_df["history_cuis"] = change_cuis(some_df.history_cuis, some_df.history_ents)
    new_df["comparison_cuis"] = change_cuis(some_df.comparison_cuis, some_df.comparison_ents)

    # put the columns in the output df into the right order
    columns = ["study", "indic_filtered", "indication_cuis", "indication_ents",
               "clinical_question", "clinical_q_cuis", "clinical_q_ents",
               "hist_finished", "history_cuis", "history_ents", "comparison", "comparison_cuis", "comparison_ents"]

    return new_df[columns]


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

nlp = spacy.load("en_core_sci_sm")

config = configparser.ConfigParser()
config.read('../../data/config/local_config.ini')

conn = sqlite3.connect(config['DATABASE']['path'])

df_whole_reports = pd.read_sql("select * from mimic_cxr_reports", conn)
df_sectioned_reports = pd.read_sql("select * from mimic_cxr_sectioned", conn)

# the position of the "?" in identification_words is important but the order of the rest of the list or the len can be changed
identification_words = ["?", "eval", "to rule out", "rule out", "assess", "infiltrate", "questionable", "followup",
                        "follow up", "follow-up", "please", "pls", "reassess"]

nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
# resolve_abbreviations=True is ignored if there is no AbbreviationDetector in the pipeline

linker = nlp.get_pipe("scispacy_linker")

# applying everything and saving parts
# split the data into 11 parts and save each part in drive after processing
how_many_parts_finished = 0

section_is = [25000 * i for i in range(how_many_parts_finished, 10)]  # 25000
section_is += [len(df_sectioned_reports)]

for n in range(len(section_is) - 1):
    new_start_i = section_is[n]
    new_end_i = section_is[n + 1]

    goal_df_name = f"goal_df_i_{new_start_i}_untill_{new_end_i}"
    print(f"\nWorking on: {goal_df_name}")

    input_df = df_sectioned_reports[new_start_i:new_end_i]
    goal_df_no_cuis = create_goal_df_no_cuis(input_df)
    goal_df_cuis = edit_goal_df_cuis(goal_df_no_cuis)
    goal_df_cuis.to_json(goal_df_name + ".json",
                         default_handler=str)  # reccursion error without the default_handler=str
    goal_df_cuis.to_csv(goal_df_name + ".txt", sep=str(
        "🎃"))  # the pumpkin is used as a seperator because normal seperators could be in the notes and mess up the table

complete_goal_df_cuis_txt = combine_and_save("txt")
complete_goal_df_cuis_withEmptyStrings_json = combine_and_save("json")
empty_str_mask = complete_goal_df_cuis_withEmptyStrings_json == ""
complete_goal_df_cuis_json_wo_empty_str = complete_goal_df_cuis_withEmptyStrings_json.copy()
complete_goal_df_cuis_json_wo_empty_str[empty_str_mask] = np.nan
complete_goal_df_cuis_json_wo_empty_str.to_excel("complete_goal_df_cuis_fromCLEAN_V4_3.xlsx")

lookup_set = create_lookup_set("complete_goal_df_cuis_fromCLEAN_V4_3.txt")
lookup_table = create_lookup_table(lookup_set)
lookup_table_dict = dict([(cui, name) for cui, name in lookup_table.values])

# Changing male/female data in final output
# The de-identification leads to a lot of "___M" & "___F" in the notes -> the entity linker classifies them not as
# male & female but as 'phenylalanine'& 'methionine'.
# Therefore, I manually change this in the goal_df.
complete_goal_df_cuis_json_fm = apply_change_cuis(complete_goal_df_cuis_json_wo_empty_str)

# store final output
complete_goal_df_cuis_json_fm.to_json("complete_goal_df_cuis_fromCLEAN_V4_4" + ".json", default_handler=str)
