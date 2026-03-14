# cs229_tcm

Manual Data Processing
eight_principles_terms.csv: directly taken from the symptom definitions from the original dataset Synptoms_Data_SymMap_SMTS.xlsx

Data Processing Flow
-> cs229RCMdatasetScraper.py 
-> cs229TCMSyndromeMulti_Hot.py
-> cs229TCMLocation.py, label_smts_by_id.py
-> cs229TCMConceptRoller.py

Output (Datasets)
- Final_Training_Features_Syndrome_Symptom.csv
- SMTS_eight_principles_by_id.csv
- Symptom_Syndrome_Edges.csv
- Syndrome_Herb_Edges.csv

Testing Data
- Synthetic_Patient_Labels.csv
- Synthetic_Patient_Symptoms.csv

Baseline model: TAN
Real model: Neural Network