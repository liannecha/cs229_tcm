# cs229_tcm



Data Processing Flow
-> cs229RCMdatasetScraper.py 
-> cs229TCMSyndromeMulti_Hot.py
-> cs229TCMLocation.py, symptoms_eight_principles_multi_hot.py
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

----------

With Herb Layer:
Additional Data Processing
herbs_eight_principles_multi_hot.py 
-> Herb_Eight_Principles_Multihot.csv