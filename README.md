# Causal Insights from Clinical Information in Radiology: Enhancing Future Multimodal AI Development

---
![MIMIC Workflow](data/viz/teaser.png)
This repository contains the code for the paper:  
**"Causal Insights from Clinical Information in Radiology: Enhancing Future Multimodal AI Development"**  
Submitted to the journal **Computer Methods and Programs in Biomedicine**.

---

## 🚀 MIMIC Dataset Preprocessing

Before starting, **request access to the dataset for the appropriate cloud platform via the PhysioNet project page**.  
Further instructions are available on the **[MIMIC website](https://mimic.mit.edu/)**.

Init a new sqlite table and add config to `local_config.ini`. 
Download and import the following MIMIC data sources into the sqlite database:  
🔹 **mimic-cxr-2.0.0-metadata**  
🔹 **edstays**  
🔹 **admissions**  
🔹 **mimic-cxr-2.0.0-chexpert**  

### 📍 Step 1: Generate Patient Timeline Table  
📌 **Script:** `prepro/1_patient_timeline_generation.py`  
🔹 Curates the `tmp_timeline` table using:  
   - `mimic-cxr-2.0.0-metadata`  
   - `edstays`  
   - `admissions`  

### 📍 Step 2: Construct CXR Timeline Table  
📌 **Script:** `prepro/2_patient_cxr_timeline_generation.py`  
🔹 Generates the `streamlit_timeline_data` table based on:  
   - `tmp_timeline`  
   - `mimic-cxr-2.0.0-chexpert`  
🔹 This table serves as the foundation for further analysis.  

### 📍 Step 3: Extract Indication Section  
📌 **Script:** `prepro/3_create_section_files_indication.py`  
🔹 Extracts the **indication** section from reports and stores it as a `.csv` file.  
🔹 Should be imported into **SQLite** as a separate table: `mimic_cxr_sectioned`.  

### 📍 Step 4: Question Extraction & Entity Linking  
📌 **Script:** `prepro/4_question_extraction_and_el.py`  
🔹 Extracts **questions** from the indication section.  
🔹 Normalizes medical concepts by:  
   1️⃣ Extracting from **indication, history, and comparison** sections.  
   2️⃣ Linking them to **UMLS concepts** via **scispaCy entity linker**.  
🔹 Output is stored as a `.json` file and should be imported into **SQLite** as `referral_information`.  

🔹 **At this point, all necessary tables are prepared for causal analysis.** 🎯  

---

## 🔬 Causal Analysis

The **notebook** `src/propensity_score_matching` guides you through the full causal analysis process.  
📌 **Simply execute the cells iteratively to proceed.**  

---

## 📊 Streamlit Dashboard

📌 **Script:** `streamlit_dashboard/app.py`  
🔹 Provides an interactive **dashboard** for exploratory analysis of the preprocessed SQLite tables.
