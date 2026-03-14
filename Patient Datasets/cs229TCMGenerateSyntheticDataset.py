"""
Owner: Ethan

Generates a synthetic dataset of patients using a context aware strategy, 
where we use the compatibility scores between symptoms and concepts to guide 
the generation of new symptoms for each syndrome. This ensures that the synthetic patients 
we generate are plausible and consistent with the underlying TCM theory, rather than 
just randomly adding symptoms which may not make sense together. 

We also drop some existing symptoms with a certain probability to make it more 
realistic and less likely to just learn the original dataset. The resulting synthetic 
dataset can be used to train our model and improve its generalisation performance.
"""
# Generate a dataset of synthetic patients
# using context aware strategy to ensure plausible patient conditions

import pandas as pd
import numpy as np

def generate_aware_patients(num_patients_per_syndrome=25, drop_prob=0.25, num_extra_symptoms=3):
    # load data
    X_symptoms = pd.read_csv("Final_Training_Features_Syndrome_Symptom.csv", index_col=0)
    X_symptoms.columns = X_symptoms.columns.astype(int) # set to int for easier indexing

    syndrome_concepts = pd.read_csv("Syndrome_Concept_Targets.csv", index_col=0)

    # rebuild df_combined_kangae
    df_locations = pd.read_csv("Symptom_Location_Features.csv", index_col='TCM_symptom_id')
    df_coords = pd.read_csv("SMTS_eight_principles_by_id.csv", index_col='TCM_symptom_id')
    df_combined_kangae = pd.concat([df_locations, df_coords], axis=1)
    df_combined_kangae = df_combined_kangae.reindex(X_symptoms.columns).fillna(0)

    # convert to numpy for easier manipulation
    symptom_concept_matrix = df_combined_kangae.values

    synthetic_patients = []
    synthetic_labels = []

    for syndrome_id in range(len(X_symptoms)):
        # these are the symptoms associated with the syndrome
        # the "knowledge graph" that we are training the model on
        set_symptoms = X_symptoms.iloc[syndrome_id].values

        # these are the concepts associated with the syndrome
        set_concepts = syndrome_concepts.iloc[syndrome_id].values

        # now we check compatibility scores between the symptoms and concepts
        # this is the "context aware" part, where we want to ensure that the synthetic 
        # patients we generate are plausible given the underlying TCM theory
        compatibility_scores = symptom_concept_matrix.dot(set_concepts)

        # make sure the model doesn't add the same symptoms
        compatibility_scores[set_symptoms == 1] = 0

        # now we can make probability scores, normalise compatibility scores with the probability distribution
        if compatibility_scores.sum() > 0:
            prob_scores = compatibility_scores / compatibility_scores.sum()
        else:
            # this is a problem--if there is a syndrom with no associated concepts
            # will lead to a divide by 0 case, so should just make it a uniform distribution
            # replace with ones and then normalise
            prob_scores = np.ones_like(compatibility_scores) / len(compatibility_scores)
            prob_scores[set_symptoms == 1] = 0 # still need to make sure we don't add the same symptoms
            prob_scores = prob_scores / prob_scores.sum() # re-normalise after zeroing out the existing symptoms

        # now we generate our synthetic patients by sampling from the probability distribution
        for patient in range(num_patients_per_syndrome):
            patient_symptoms = set_symptoms.copy()
            # we will first drop some existing symptoms with a certain probability, to make it more realistic and 
            # less likely to just learn the original dataset
            # find the symptoms that are currently active
            active_symptoms = np.where(patient_symptoms == 1)[0]
            # randomly drop some of the active symptoms
            drop_num = int(len(active_symptoms) * drop_prob)
            if drop_num > 0:
                # drop symptoms by setting them to 0
                drop_symptoms = np.random.choice(active_symptoms, size=drop_num, replace=False)
                patient_symptoms[drop_symptoms] = 0

            # then we add some new symptoms based on the probability distribution
            # these are context aware additions, using the compatibility scores
            new_symptoms = np.random.choice(
                len(patient_symptoms),
                size=num_extra_symptoms,
                replace=False,
                p=prob_scores # use the probability scores to sample new symptoms, so more compatible symptoms are more likely to be added
            )
            patient_symptoms[new_symptoms] = 1 # set the new symptoms to 1

            synthetic_patients.append(patient_symptoms)
            synthetic_labels.append(syndrome_id)

    df_synthetic_X = pd.DataFrame(synthetic_patients, columns=X_symptoms.columns)
    df_synthetic_y = pd.DataFrame(synthetic_labels, columns=['Syndrome_id'])

    df_synthetic_X.to_csv("Synthetic_Patient_Symptoms.csv", index=False)
    df_synthetic_y.to_csv("Synthetic_Patient_Labels.csv", index=False)

    print("Did It!")
    return df_synthetic_X, df_synthetic_y

synthetic_X, synthetic_y = generate_aware_patients(
    num_patients_per_syndrome=25, # this will give us a total of 25 * 233 = 5825 synthetic patients, which is a decent size for training
    drop_prob=0.25, # we will drop 25% of the existing symptoms to make it more realistic and less likely to just learn the original dataset
    num_extra_symptoms=3 # we will add 3 new symptoms based on the compatibility scores, to make it more context aware and plausible
)