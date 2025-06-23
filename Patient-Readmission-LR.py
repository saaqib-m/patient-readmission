import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.feature_extraction import text 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import re 
from rapidfuzz import process, fuzz

df_admits = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\admissions.csv.gz', compression='gzip')
df_pat = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\patients.csv.gz', compression='gzip')
df_icd_proc = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\d_icd_procedures.csv.gz', compression='gzip')
df_icd_diagnosis = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\d_icd_diagnoses.csv.gz', compression='gzip')
df_proc = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\procedures_icd.csv.gz', compression='gzip')
df_diagnosis = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\diagnoses_icd.csv.gz', compression='gzip')
df_medi = pd.read_csv(r'C:\Users\Saaqib\Documents\Projects\MIMIC_IV\emar.csv.gz', compression='gzip')

# Join diagnosis, admissions and procs
df_admit_icd = pd.merge(df_admits,df_diagnosis,how = 'left', on = 'hadm_id')
df_admit_diagnosis = pd.merge(df_admit_icd,df_icd_diagnosis,how = 'left', on = 'icd_code')
df_admit_diagnosis = pd.merge(df_admit_diagnosis,df_icd_proc,how = 'left', on = 'icd_code')
df_admit_diagnosis.drop(columns= ['subject_id_y'], inplace = True)
df_admit_diagnosis.rename(columns={'long_title_x': 'icd_diagnosis_name','long_title_y':'proc_name','subject_id_x':'subject_id'}, inplace=True)
df_admit_diagnosis = pd.merge(df_admit_diagnosis,df_pat[['subject_id', 'anchor_age','gender']],how = 'left', on = 'subject_id')

# Replace all NaN
df_admit_diagnosis['icd_diagnosis_name'] = df_admit_diagnosis['icd_diagnosis_name'].fillna('No Diagnosis')
df_admit_diagnosis['proc_name'] = df_admit_diagnosis['proc_name'].fillna('No Procedure')
df_admit_diagnosis['icd_code'] = df_admit_diagnosis['icd_code'].fillna('No ICD Code')

my_stop_words = ['term','episode', 'face', 'facial','mouth','home','legs','pain'
                 ,'procedure','procedures','diagnosis','weight','adult','child','unspecified','primary','secondary','other','long'
                 ,'system', 'due', 'related', 'associated', 'organ', 'tissue', 'area', 'region', 'tract'
                 ,'side', 'lobe', 'part', 'left', 'right', 'bilateral', 'upper', 'lower', 'anterior','encounter'
                 ,'posterior', 'icd', 'code','type', 'specified', 'classification', 'negative','positive'
                 ,'history', 'present', 'past','recent', 'new', 'old','accessory','adhesive','body', 'blood'
                 ,'classified','unclassified','cause','causing','reaction','status','current','complication'
                 ,'do','not','classified','elsewhere','severity','without','history','personal','cause','initial']
combined_stop_words = list(text.ENGLISH_STOP_WORDS.union(my_stop_words))

def normalise_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    tokens = text.split()                    # Tokenize
    filtered_tokens = [word for word in tokens if word not in combined_stop_words]  # Remove stop words
    return ' '.join(filtered_tokens)

def standardise_similar_entries(unique_values, threshold=80):
    standardisation_map = {}
    value_list = list(unique_values)
    
    for i, current_value in enumerate(value_list):
        if current_value is None or current_value == '':
            continue  # skip empty
        
        if current_value in standardisation_map:
            continue  # already mapped
        
        # Create a list excluding current_value
        others = value_list[:i] + value_list[i+1:]
        
        match = process.extractOne(
            current_value,
            others,
            processor=None,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold
        )
        
        if match:
            matched_value = match[0]
            standardisation_map[matched_value] = current_value
            standardisation_map[current_value] = current_value
        else:
            # map self if no close match found
            standardisation_map[current_value] = current_value
            
    return standardisation_map

df_admit_diagnosis['diagnosis_clean'] = df_admit_diagnosis['icd_diagnosis_name'].apply(normalise_text)
df_admit_diagnosis['procedure_clean'] = df_admit_diagnosis['proc_name'].apply(normalise_text)
unique_diagnoses = df_admit_diagnosis['diagnosis_clean'].dropna().unique()
unique_proc = df_admit_diagnosis['procedure_clean'].dropna().unique()

# Map the cleaned diagnosis to the standardised form
standardised_map_diagnosis = standardise_similar_entries(unique_diagnoses)
standardised_map_proc = standardise_similar_entries(unique_proc)
df_admit_diagnosis['diagnosis_name'] = df_admit_diagnosis['diagnosis_clean'].map(standardised_map_diagnosis).fillna(df_admit_diagnosis['icd_diagnosis_name'])
df_admit_diagnosis['procedure_name'] = df_admit_diagnosis['procedure_clean'].map(standardised_map_proc).fillna(df_admit_diagnosis['proc_name'])

diagnosis_counts = df_admit_diagnosis.diagnosis_name.value_counts()
proc_counts = df_admit_diagnosis.procedure_name.value_counts()

# Get the top 200 by frequency
top_200_diagnosis = df_admit_diagnosis['diagnosis_name'].value_counts().nlargest(201).index
top_200_proc = df_admit_diagnosis['procedure_name'].value_counts().nlargest(201).index

# Map ICD codes — keep top 200, replace others with 'Other Code'
df_admit_diagnosis['diagnosis_converted'] = df_admit_diagnosis['diagnosis_name'].apply(
    lambda x: str(x) if x in top_200_diagnosis else 'Other Diagnosis')
df_admit_diagnosis['procedure_converted'] = df_admit_diagnosis['procedure_name'].apply(
    lambda x: str(x) if x in top_200_proc else 'Other Procedure')
df_admit_diagnosis.drop(columns = ['icd_version_y','icd_version_x','edregtime','edouttime','deathtime','admit_provider_id','hospital_expire_flag','icd_diagnosis_name','proc_name',
                                         'seq_num','icd_code','language','icd_version','diagnosis_clean','diagnosis_name','procedure_clean','procedure_name'],inplace = True)

# Group first to reduce data size
df_admit_diagnosis_grouped = df_admit_diagnosis.groupby(['hadm_id', 'subject_id']).agg({  
     'admission_type':'first',
     'admission_location': 'first',
     'discharge_location':'first',
     'admittime':'first',
     'dischtime':'first',
     'insurance':'first',
     'marital_status': 'first',
     'race':'first',
     'anchor_age': 'first',
     'gender': lambda x: set(x),
     'diagnosis_converted': lambda x: set(x),
     'procedure_converted': lambda x: set(x),
     }).reset_index()

# Assign admission ranks throughout time
# Readmission is within 30 days of discharge and next admission so insert flag
df_admit_diagnosis_grouped['admittime'] = pd.to_datetime(df_admit_diagnosis_grouped['admittime'])
df_admit_diagnosis_grouped['dischtime'] = pd.to_datetime(df_admit_diagnosis_grouped['dischtime'])

df_admit_diagnosis_grouped['admission_rank'] = df_admit_diagnosis_grouped.groupby('subject_id')['admittime'].rank(method='dense', ascending=True)
df_admit_diagnosis_grouped.sort_values(by = ['admission_rank'],ascending= True, inplace= True)
# Get next admission's out time for each patient
df_admit_diagnosis_grouped['last_dischtime'] = df_admit_diagnosis_grouped.groupby('subject_id')['dischtime'].shift(1)

# Calculate days until next visit
df_admit_diagnosis_grouped['days_from_last'] = (df_admit_diagnosis_grouped['admittime'] - df_admit_diagnosis_grouped['last_dischtime']).dt.days

# Create readmission flag (1 if last visit is within 30 days, else 0)
df_admit_diagnosis_grouped['readmission'] = df_admit_diagnosis_grouped['days_from_last'].apply(lambda x: 1 if pd.notnull(x) and x <= 30 and x >= 0 else 0)

# one hot encode the text columns
df_admit_diagnosis_encoded = df_admit_diagnosis_grouped.drop(columns= ['admittime','dischtime','last_dischtime','days_from_last', 
                                                                       'admission_rank']) # drop irrelevant cols

cols_to_encode = ['admission_type','admission_location','discharge_location','insurance','race','marital_status']
df_admit_diagnosis_encoded = pd.get_dummies(df_admit_diagnosis_encoded, columns=cols_to_encode, drop_first=True,dtype=int)

for col in ['diagnosis_converted', 'procedure_converted','gender']:
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(df_admit_diagnosis_encoded[col])
    df_encoded = pd.DataFrame.sparse.from_spmatrix(
        X,
        index=df_admit_diagnosis_encoded['hadm_id'],
        columns=[f"{c}" for c in mlb.classes_]
    )
    df_admit_diagnosis_encoded = df_admit_diagnosis_encoded.merge(
        df_encoded, left_on='hadm_id', right_index=True, how='left'
    )

df_admit_diagnosis_encoded.drop(columns=['diagnosis_converted','procedure_converted','gender'], inplace= True)

# Medication Data
df_medi.drop(columns = ['enter_provider_id','charttime','scheduletime','storetime','emar_id','emar_seq','poe_id','pharmacy_id','enter_provider_id'
                        ,'charttime','scheduletime','storetime'], inplace= True)
df_medi['medication'] = df_medi['medication'].fillna('No Medication')

# Define grouping rules for usage
administered = [
    'Administered', 'Partial Administered', 'Applied',
    'Confirmed', 'Started', 'Administered in Other Location',
    'Started in Other Location', 'Administered Bolus from IV Drip',
    'Applied in Other Location', 'Removed Existing / Applied New',
    'Removed Existing / Applied New in Other Location',
    'Restarted', 'Restarted in Other Location'
]

not_administered = [
    'Not Applied', 'Not Given', 'Not Flushed', 'Hold Dose', 'Removed',
    'Not Confirmed', 'Not Started', 'Not Assessed', 'Not Removed',
    'Not Stopped', 'Not Read', 'Not Given per Sliding Scale',
    'Not Stopped per Sliding Scale', 'Not Started per Sliding Scale',
    'Not Given per Sliding Scale in Other Location', 'Read', 'Read in Other Location', 'Stopped', 'Stopped As Directed',
    'Stopped - Unscheduled', 'Stopped - Unscheduled in Other Location',
    'Rate Change', 'Rate Change in Other Location','Removed in Other Location',
    'Confirmed in Other Location'

]

delayed = [s for s in df_medi['event_txt'].dropna().unique() if s.startswith('Delayed')]

other = [
    'Flushed', 'Flushed in Other Location', 'Assessed',
    'Assessed in Other Location', 'Infusion Reconciliation',
    'Infusion Reconciliation Not Done', 'Pain score re-assessment',
    'Pain score re-assess not done', 'Documented in O.R. Holding',
    'TPN Rate Not Changed', ' in Other Location', 'Partial ',
]

# Make sure event lists are sets for faster lookup
administered_set = set(administered)
not_administered_set = set(not_administered)
delayed_set = set(delayed)

# Initialize with 'Other'
df_medi['usage'] = 'Other'

# Apply conditions using boolean indexing
mask_admin = df_medi['event_txt'].isin(administered_set) & (df_medi['medication'] != 'No Medication')
mask_not_admin = df_medi['event_txt'].isin(not_administered_set)
mask_delayed = df_medi['event_txt'].isin(delayed_set)

df_medi.loc[mask_admin, 'usage'] = 'Administered'
df_medi.loc[mask_not_admin, 'usage'] = 'Not Administered'
df_medi.loc[mask_delayed, 'usage'] = 'Delayed'

# Filter to only include 'Administered'
df_medi = df_medi[df_medi['usage'] == 'Administered']
df_medi['medication'] = df_medi['medication'].str.lower()

df_medi[df_medi['usage']=='Administered'] # only administered values will have an affect
top_100_meds = df_medi['medication'].value_counts().nlargest(201).index

df_medi['medication_grouped'] = df_medi['medication'].apply(
    lambda x: str(x) if x in top_100_meds else 'Other Medication'
)
# Group first to reduce data size
df_medi.drop(columns=['medication','event_txt','usage'],inplace=True)

df_medi_grouped = df_medi.groupby(['hadm_id', 'subject_id']).agg({  
     'medication_grouped': lambda x: set(x),
     }).reset_index()

mlb = MultiLabelBinarizer(sparse_output=True)
X = mlb.fit_transform(df_medi_grouped['medication_grouped'])

df_medi_encoded = pd.DataFrame.sparse.from_spmatrix(
    X,
    index=df_medi_grouped['hadm_id'],
    columns=mlb.classes_
)
df_medi_encoded.drop(columns=['Other Medication'], inplace=True)

df_full_dataset = pd.merge(
    df_admit_diagnosis_encoded,
    df_medi_encoded,
    how='left',
    on='hadm_id'
)
df_full_dataset.fillna(0, inplace=True)

# Prepare Model
X = df_full_dataset.drop('readmission', axis=1)
y = df_full_dataset.readmission

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
logreg = LogisticRegression(penalty= 'l1',solver='liblinear', class_weight = {0: 1, 1: 5}, random_state=42, max_iter=1000)

# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
coef = logreg.coef_[0]  # For binary classification
feature_names = X_train.columns  

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coef,
    'Odds Ratio': np.exp(coef),
    'Diagnosis Name': "",
    'Procedure Name': "",
    "Medication Name": ""
})

# Prepare lookup tables
diagnosis_lookup = df_admit_diagnosis['diagnosis_converted'].unique()
proc_lookup = df_admit_diagnosis['procedure_converted'].unique()
diagnosis_lookup = diagnosis_lookup[diagnosis_lookup != 'Other Diagnosis']
proc_lookup = proc_lookup[proc_lookup != 'Other Procedure']
medication_lookup = df_medi['medication_grouped'].unique()  

# Loop through each row
for idx, row in coef_df.iterrows():
    feature = row['Feature']

    if feature in diagnosis_lookup:
        coef_df.at[idx, 'Diagnosis Name'] = feature
    
    if feature in proc_lookup:
        coef_df.at[idx, 'Procedure Name'] = feature

    # If it's a medication, keep the name
    if feature in medication_lookup:
        coef_df.at[idx, 'Medication Name'] = feature

print(coef_df.head())