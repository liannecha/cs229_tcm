"""
Owner: Ethan and Lianne

Same as cs229TCMSyntheticNeuralNetwork.py but with additional herb head.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import os


# Builds a look up table for herb -> concept vector (14 dim)
# need to define this before the model because we pass the matrix into TCMNet as a constructor argument
def build_herb_concept_matrix(location_file, principles_file, smhb_excel_file, syndrome_herb_file):
    # concept column order must match the model's output
    concept_order = ['Wood', 'Fire', 'Earth', 'Metal', 'Water', 'Reproductive',
                     'Hot', 'Cold', 'Internal', 'External', 'Deficiency', 'Excess', 'Yin', 'Yang']

    # load location features (matches Syndrome_Herb_Targets columns)
    loc_df = pd.read_csv(location_file)
    loc_df = loc_df.set_index('Herb_id')
    loc_df = loc_df[~loc_df.index.duplicated(keep='first')] # drop duplicate herb ids from exploded multi-herb rows
    loc_df.columns = ['Wood', 'Fire', 'Earth', 'Metal', 'Water', 'Reproductive'] # rename to match concept order

    # load eight principles features; indexed by HERBDB_ID
    prin_df = pd.read_csv(principles_file)
    prin_df = prin_df.set_index('HERBDB_ID')
    # reorder columns to match concept order
    prin_df = prin_df[['hot', 'cold', 'internal', 'external', 'deficiency', 'excess', 'yin', 'yang']]
    prin_df.columns = ['Hot', 'Cold', 'Internal', 'External', 'Deficiency', 'Excess', 'Yin', 'Yang']

    # load the original excel to get the Herb_id to HERBDB_ID mapping
    # need for one-to-many mapping
    excel_df = pd.read_excel(smhb_excel_file)[['Herb_id', 'HERBDB_ID']].dropna()
    excel_df['HERBDB_ID'] = excel_df['HERBDB_ID'].astype(str).str.split('|')
    excel_df = excel_df.explode('HERBDB_ID')
    excel_df['HERBDB_ID'] = excel_df['HERBDB_ID'].str.strip()
    excel_df['Herb_id'] = excel_df['Herb_id'].astype(int)
    id_map = excel_df.set_index('HERBDB_ID')['Herb_id']

    # map eight principles onto Herb_id to combine with location features
    prin_df['Herb_id'] = prin_df.index.map(id_map)
    prin_df = prin_df.dropna(subset=['Herb_id'])
    prin_df['Herb_id'] = prin_df['Herb_id'].astype(int)
    prin_df = prin_df.set_index('Herb_id')
    # drop duplicate herb ids after mapping
    prin_df = prin_df[~prin_df.index.duplicated(keep='first')]

    # get the herb ids that actually appear in Syndrome_Herb_Targets so matrix columns align
    # columns are SMHB format strings (e.g. SMHB00174); numeric part matches Excel Herb_id (174)
    syndrome_herb_df = pd.read_csv(syndrome_herb_file, index_col=0)
    herb_ids = syndrome_herb_df.columns.tolist()  # keep as SMHB strings for output
    herb_ids_numeric = [int(h.replace('SMHB', '')) for h in herb_ids]  # numeric part for reindexing feature files

    # build final matrix: reindex both feature sets to the syndrome-herb herb ids, fill missing with 0
    loc_aligned = loc_df.reindex(herb_ids_numeric).fillna(0)
    prin_aligned = prin_df.reindex(herb_ids_numeric).fillna(0)

    # concatenate location and principles into one matrix
    herb_concept_matrix = pd.concat([loc_aligned, prin_aligned], axis=1)[concept_order].values

    return herb_concept_matrix, herb_ids


class SyntheticTCMDataset(Dataset):
    def __init__(self, synthetic_X_file, synthetic_y_file, concept_file, syndrome_herb_file=None):
        # load features and targets
        self.X_df = pd.read_csv(synthetic_X_file)
        self.y_df = pd.read_csv(synthetic_y_file)
        self.concept_df = pd.read_csv(concept_file, index_col=0)

        # convert to tensors
        self.X = torch.tensor(self.X_df.values, dtype=torch.float)
        # classification labels for syndromes, need to be long for cross-entropy loss
        self.y_syndrome = torch.tensor(self.y_df['Syndrome_id'].values, dtype=torch.long)

        self.syndrome_names = [f"Syndrome_{i}" for i in self.y_syndrome.numpy()] # just for plotting later

        # map concepts to patients
        mapped_concepts = self.concept_df.values[self.y_syndrome.numpy()]
        self.concept_targets = torch.tensor(mapped_concepts, dtype=torch.float)

        # map herb multi-hot targets to patients via their syndrome
        # syndrome_herb_file has one row per syndrome and one column per herb
        # so we can look up each patient's herb targets using their syndrome id, same way we did for concepts
        if syndrome_herb_file is not None:
            syndrome_order = self.concept_df.index
            herb_df = pd.read_csv(syndrome_herb_file, index_col=0).reindex(syndrome_order).fillna(0)
            self.herb_ids = herb_df.columns.tolist()
            mapped_herbs = herb_df.values[self.y_syndrome.numpy()]
            self.herb_targets = torch.tensor(mapped_herbs, dtype=torch.float)
        else:
            self.herb_ids = []
            self.herb_targets = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.herb_targets is not None:
            return self.X[idx], self.concept_targets[idx], self.y_syndrome[idx], self.herb_targets[idx]
        return self.X[idx], self.concept_targets[idx], self.y_syndrome[idx]

class TCMNet(nn.Module):
    def __init__(self, num_symptoms=1717, num_concepts=14, num_syndromes=233, herb_concept_matrix=None):
        super(TCMNet, self).__init__()

        # feature extraction layers
        # doing a sequential structure so that model will learn coord and location features instead of simple syndrome matching
        # sequential for the shared layer so that the model will understand complex system interactions
        # symptoms are highly non-linear in tcm
        self.shared_layer = nn.Sequential(
            nn.Linear(num_symptoms, 512), # draw basic connections
            nn.ReLU(), # non-linear activation
            nn.Dropout(0.3) # dropout to prevent overfitting given small dataset
        )

        # concept prediction head
        # Linear to translate from shared features to concept percentages
        # project 512 deep features into the 14 buckets
        self.concept_head = nn.Linear(512, num_concepts)

        # syndrome classification head
        # need sequential to combine the shared features with the concept predictions,
        # so that the model can learn to use the concepts to predict the syndrome
        # Input is shared features + concept predictions, to encourage learning of concepts
        self.syndrome_head = nn.Sequential(
            nn.Linear(512 + num_concepts, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_syndromes)
        )

        if herb_concept_matrix is not None:
            # register the herb concept matrix as a fixed buffer inside the model
            # this means it gets saved with the model and moves to gpu automatically, but it doesn't get trained
            # shape is (num_herbs, 14) where each row is that herb's location + eight principles features
            self.register_buffer('herb_concept_mat', torch.tensor(herb_concept_matrix, dtype=torch.float))

            # herb scorer: for each herb, concatenate everything we know about the patient with that herb's known concept features
            # and score it with a small mlp
            # input is shared_features (512) + concept_preds (14) + syndrome_probs (233) + herb_concept_features (14) = 773 dims
            # we feed everything in directly so no information is lost before scoring
            # we apply this to all herbs at once using broadcasting so we don't need a loop
            self.herb_scorer = nn.Sequential(
                nn.Linear(512 + num_concepts + num_syndromes + num_concepts, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        else:
            self.herb_concept_mat = None
            self.herb_scorer = None

    # need a forward pass
    def forward(self, x):
        # extract deep features
        shared_features = self.shared_layer(x)

        # predict concepts from shared features
        concept_preds = torch.sigmoid(self.concept_head(shared_features)) # sigmoid for multi-label output

        # concatenate shared features with concept predictions for syndrome classification
        combined_features = torch.cat((shared_features, concept_preds), dim=1)

        # make final syndrome prediction
        syndrome_preds = self.syndrome_head(combined_features)

        if self.herb_concept_mat is not None:
            # concatenate everything we know about the patient into one vector
            # we use softmax syndrome probs (not argmax) so uncertainty over syndromes flows through to herbs
            syndrome_probs = torch.softmax(syndrome_preds, dim=1)
            patient_context = torch.cat((shared_features, concept_preds, syndrome_probs), dim=1)  # (batch, 759)

            batch_size = patient_context.size(0)
            num_herbs = self.herb_concept_mat.size(0)

            # broadcast patient context and herb features so we can score all herbs at once
            patient_context_exp = patient_context.unsqueeze(1).expand(-1, num_herbs, -1)    # (batch, num_herbs, 759)
            herb_feats_exp = self.herb_concept_mat.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_herbs, 14)

            # concatenate and score: (batch, num_herbs, 773) -> (batch, num_herbs)
            scorer_input = torch.cat([patient_context_exp, herb_feats_exp], dim=2)
            herb_preds = torch.sigmoid(self.herb_scorer(scorer_input).squeeze(2)) # sigmoid for multi-label

            return concept_preds, syndrome_preds, herb_preds

        return concept_preds, syndrome_preds

# now training loop
def train_model(model, dataloader, epochs=100, class_weights=None, lambda_concept=1.0, lambda_syndrome=1.0, lambda_herb=1.0):
    # We are going to use mean squared error for the concept predictions, and cross-entropy loss for the syndrome classification
    # the issue with having two different loss functions is that they are on different scales
    # this is leading to an oversmearing problem, so we can just scale the losses to be on a similar scale

    # another fix to the oversmearing problem is to overly punish rare symptoms
    # that way, the model doesn't overpredict common symptoms and underpredict rare symptoms, which is what we are seeing right now
    # we can do this by adding a weight to the MSE loss for each symptom, based on the inverse frequency of that symptom in the dataset

    criterion_concept = nn.MSELoss() # mse for distance, since its continuous values

    if class_weights is not None:
        # move weights here
        # need to ensure class_weights is a tensor on the same device as the model
        criterion_syndrome = nn.CrossEntropyLoss(weight=class_weights) # cross-entropy for classification, since we are choosing one category, need to punish
    else:
        criterion_syndrome = nn.CrossEntropyLoss()

    # bce for herb prediction since each herb is an independent binary decision (patient can use multiple herbs)
    criterion_herb = nn.BCELoss()

    has_herb_head = model.herb_concept_mat is not None

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'total': [], 'concept': [], 'syndrome': [], 'herb': []}

    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_syndrome = 0.0
        epoch_concept = 0.0
        epoch_herb = 0.0

        for batch in dataloader:
            optimizer.zero_grad() # zero the parameter gradients

            if has_herb_head:
                symptoms, true_concepts, true_syndromes, true_herbs = batch
                # forward pass
                pred_concepts, pred_syndromes, pred_herbs = model(symptoms.float())
            else:
                symptoms, true_concepts, true_syndromes = batch
                # forward pass
                pred_concepts, pred_syndromes = model(symptoms.float())

            true_concepts = true_concepts.float() # ensure true concepts are float for MSE
            true_syndromes = true_syndromes.long() # ensure true syndromes are long for cross-entropy

            # calculate losses
            # here is where we can scale the losses
            raw_loss_concept = criterion_concept(pred_concepts, true_concepts)
            raw_loss_syndrome = criterion_syndrome(pred_syndromes, true_syndromes)

            # scale the losses to be on a similar scale
            scaled_loss_concept = raw_loss_concept * lambda_concept
            scaled_loss_syndrome = raw_loss_syndrome * lambda_syndrome

            # combine for total loss
            loss = scaled_loss_concept + scaled_loss_syndrome

            if has_herb_head:
                true_herbs = true_herbs.float()
                raw_loss_herb = criterion_herb(pred_herbs, true_herbs)
                loss = loss + raw_loss_herb * lambda_herb
                epoch_herb += raw_loss_herb.item()

            # backprop
            loss.backward()
            optimizer.step()

            # we track raw losses for plotting
            epoch_total += loss.item()
            epoch_concept += raw_loss_concept.item()
            epoch_syndrome += raw_loss_syndrome.item()

        history['total'].append(epoch_total / len(dataloader))
        history['concept'].append(epoch_concept / len(dataloader))
        history['syndrome'].append(epoch_syndrome / len(dataloader))
        history['herb'].append(epoch_herb / len(dataloader))

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {history['total'][-1]:.4f}")

    return history



# Now we initialise and train the model
synthetic_x_file = "Patient Datasets/Synthetic_Patient_Symptoms.csv"
synthetic_y_file = "Patient Datasets/Synthetic_Patient_Labels.csv"
concept_file = "Syndrome_Concept_Targets.csv"
syndrome_herb_file = "Processed Datasets/Syndrome_Herb_Targets.csv"

# build herb concept matrix before creating the dataset and model
# we need this first because the dataset uses it for herb targets and the model takes it as a constructor argument
print("Building herb concept matrix...")
herb_concept_matrix, herb_ids = build_herb_concept_matrix(
    location_file="Processed Datasets/Herb_Location_Features.csv",
    principles_file="Processed Datasets/Herb_Eight_Principles_Multihot.csv",
    smhb_excel_file="Original Datasets/SymMap v2.0, SMHB file.xlsx",
    syndrome_herb_file=syndrome_herb_file
)

# Create dataset and dataloader
first_dataset = SyntheticTCMDataset(synthetic_x_file, synthetic_y_file, concept_file, syndrome_herb_file=syndrome_herb_file)
dataset_size = len(first_dataset)

# split into train and validation sets
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

print(f"Dataset loaded with {len(np.unique(first_dataset.y_syndrome))} syndromes, {first_dataset.X.shape[1]} symptoms, and {len(herb_ids)} herbs.")

# create the dataloaders
train_dataset, val_dataset = random_split(first_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# initialise the model
# pass herb_concept_matrix in so it gets registered as a fixed buffer inside the model
model = TCMNet(num_symptoms=first_dataset.X.shape[1], num_concepts=14, num_syndromes=len(np.unique(first_dataset.y_syndrome)), herb_concept_matrix=herb_concept_matrix)

# extract labels for class weight calculation
train_indices = train_dataset.indices
train_labels = first_dataset.y_syndrome[train_indices].numpy() # need to convert to numpy for sklearn function

# get the unique syndrome classes for class weight calculation
# this prevents issues if some syndromes are missing from the training set due to random split, which would cause compute_class_weight to fail
sonzai_classes = np.unique(first_dataset.y_syndrome.numpy())

# here, we will calculate the class weights based on the frequency of each syndrome in the dataset, to help with the oversmearing problem
weights = compute_class_weight(class_weight='balanced', classes=sonzai_classes, y=train_labels)

class_weight_tensor = torch.tensor(weights, dtype=torch.float)

# train the model
print("Starting training...")
loss_history = train_model(
    model,
    train_loader,
    epochs=100,
    class_weights=class_weight_tensor,
    # this is to weight the loss concepts
    # Cross-entropy or syndrome loss is usually around 0.5 to 1.0, while MSE or concept loss is around 0.01 to 0.1,
    # so we need to scale the concept loss up by about 10x to be on a similar scale
    lambda_concept = 10.0,
    lambda_syndrome = 1.0,
    # herb bce loss is usually around 0.1 to 0.3, so scale it similarly to concept loss
    lambda_herb = 5.0
)

def evaluate_model(trained_model, dataloader):
    trained_model.eval() # set model to evaluation mode

    correct_syndromes = 0
    total_samples = 0
    total_concept_error = 0.0
    total_herb_precision = 0.0
    total_herb_recall = 0.0

    criterion_concept = nn.MSELoss()
    has_herb_head = trained_model.herb_concept_mat is not None

    with torch.no_grad(): # turn off gradients for evaluation
        for batch in dataloader:
            if has_herb_head:
                symptoms, true_concepts, true_syndromes, true_herbs = batch
                # forward pass
                pred_concepts, pred_syndromes, pred_herbs = trained_model(symptoms.float())
            else:
                symptoms, true_concepts, true_syndromes = batch
                # forward pass
                pred_concepts, pred_syndromes = trained_model(symptoms.float())

            # evaluate syndrome predictions
            _, predicted_syndromes = torch.max(pred_syndromes, 1)
            correct_syndromes += (predicted_syndromes == true_syndromes).sum().item()
            total_samples += true_syndromes.size(0)

            # evaluate concept predictions
            concept_error = criterion_concept(pred_concepts, true_concepts)
            total_concept_error += concept_error.item()

            # evaluate herb predictions at top-5
            if has_herb_head:
                top5_indices = pred_herbs.topk(5, dim=1).indices
                for j in range(true_syndromes.size(0)):
                    true_set = set(true_herbs[j].nonzero(as_tuple=True)[0].tolist())
                    pred_set = set(top5_indices[j].tolist())
                    true_pos = len(pred_set & true_set)
                    total_herb_precision += true_pos / len(pred_set) if pred_set else 0
                    total_herb_recall += true_pos / len(true_set) if true_set else 0

    syndrome_accuracy = (correct_syndromes / total_samples) * 100
    avg_concept_error = total_concept_error / len(dataloader)

    print("Evaluation Results:")
    print(f"Syndrome Classification Accuracy: {syndrome_accuracy:.2f}%")
    print(f"Average Concept Prediction MSE (Lower is better): {avg_concept_error:.4f}")
    if has_herb_head:
        print(f"Herb Precision@5: {total_herb_precision / total_samples:.4f}")
        print(f"Herb Recall@5:    {total_herb_recall / total_samples:.4f}")

evaluate_model(model, val_loader) # do it on the validation set now

def plot_loss_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['total'], label='Total Loss', color='black', linewidth=2)
    plt.plot(history['concept'], label='Concept Loss (MSE)', linestyle='--')
    plt.plot(history['syndrome'], label='Syndrome Loss (Cross Entropy)', linestyle='-.')
    if any(v > 0 for v in history['herb']):
        plt.plot(history['herb'], label='Herb Loss (BCE)', linestyle=':')

    plt.title('Training Loss Curve', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.show()

plot_loss_curve(loss_history)

def plot_patient_profile(model, dataset, index=0):
    model.eval() # evaluation mode

    item = dataset[index]
    symptoms, true_concepts = item[0], item[1]
    syndrome_name = dataset.syndrome_names[index]

    with torch.no_grad():
        pred_concepts = model(symptoms.unsqueeze(0).float())[0] # just need the concept predictions

    true_vals = true_concepts.numpy()
    pred_vals = pred_concepts.squeeze(0).numpy()

    concept_labels = ['Wood', 'Fire', 'Earth', 'Metal', 'Water', 'Reproductive', 'Hot', 'Cold', 'Internal', 'External', 'Deficiency', 'Excess', 'Yin', 'Yang']

    # the thing is that each symptom appears exactly once
    # so the anti over-smearing won't work until we get a larger dataset
    x = np.arange(len(concept_labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, true_vals, width, label='True Concepts', color='steelblue', alpha=0.7)
    plt.bar(x + width/2, pred_vals, width, label='Predicted Concepts', color='darkorange', alpha=0.7)

    plt.title(f"Concept Profile for {syndrome_name}", fontsize=15)
    plt.ylabel('Concept Proportion', fontsize=12)
    plt.xticks(x, concept_labels, rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()

    plt.show()

plot_patient_profile(model, first_dataset, index=0)

def evaluate_model_extra(trained_model, dataloader):
    trained_model.eval() # set model to evaluation mode

    correct_syndromes = 0
    correct_top5 = 0
    total_samples = 0

    all_true_syndromes = []
    all_pred_syndromes = []

    has_herb_head = trained_model.herb_concept_mat is not None

    with torch.no_grad(): # turn off gradients for evaluation
        for batch in dataloader:
            if has_herb_head:
                symptoms, true_concepts, true_syndromes, true_herbs = batch
                # forward pass
                _, pred_syndromes, _ = trained_model(symptoms.float())
            else:
                symptoms, true_concepts, true_syndromes = batch
                # forward pass
                _, pred_syndromes = trained_model(symptoms.float())

            # accuracy
            _, predicted_syndromes = torch.max(pred_syndromes, 1)
            correct_syndromes += (predicted_syndromes == true_syndromes).sum().item()
            total_samples += true_syndromes.size(0)

            # For f1 calc later
            all_true_syndromes.extend(true_syndromes.cpu().numpy())
            all_pred_syndromes.extend(predicted_syndromes.cpu().numpy())

            # top-5 accuracy
            _, top5_predicted = pred_syndromes.topk(5, dim=1, largest=True, sorted=True)
            for i in range(true_syndromes.size(0)):
                if true_syndromes[i] in top5_predicted[i]:
                    correct_top5 += 1

    # final evaluation metrics
    syndrome_accuracy = (correct_syndromes / total_samples) * 100
    top5_accuracy = (correct_top5 / total_samples) * 100

    macro_f1 = f1_score(all_true_syndromes, all_pred_syndromes, average='macro') * 100

    print("Extra Evaluation Results:")
    print(f"Syndrome Classification Accuracy: {syndrome_accuracy:.2f}%")
    print(f"Top-5 Syndrome Accuracy: {top5_accuracy:.2f}%")
    print(f"Macro F1 Score: {macro_f1:.2f}%")

    return {'Syndrome Accuracy': syndrome_accuracy, 'Top-5 Accuracy': top5_accuracy, 'Macro F1 Score': macro_f1}

metrics_dict = evaluate_model_extra(model, val_loader)

# plotting the overall metrics in a bar chart for better visualization
def plot_overall_metrics(metrics_dict):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(8, 6))

    bars = plt.bar(labels, values, color=['steelblue', 'darkorange', 'seagreen'], alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Overall Model Performance Metrics', fontsize=15)
    plt.ylabel('Percentage', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

plot_overall_metrics(metrics_dict)


"""
Additional herb recommendation section

Instead of the dot product approach, we now use the herb head inside the model to recommend herbs.
The herb head scores each herb by concatenating the patient embedding with that herb's known concept
features (location + eight principles) and passing through a small mlp, so the model can learn
non-linear interactions between what the patient needs and what each herb provides.

We take the top-5 herbs by predicted probability from the herb head.
-> k = 5 (returns top 5 scores)
"""

# Evaluation

model.eval()
all_precision = []
all_recall = []

# step through one representative patient per syndrome (every 25th patient, since there are 25 per syndrome)
with torch.no_grad():
    for i in range(0, len(first_dataset), 25):
        symptoms, true_concepts, true_syndrome_idx, true_herbs = first_dataset[i]
        _, _, pred_herbs = model(symptoms.unsqueeze(0).float())

        pred_herb_vec = pred_herbs.squeeze(0).numpy()

        # top-5 by predicted probability from the herb head
        top5_indices = np.argsort(pred_herb_vec)[::-1][:5]
        recommended_herb_set = set(first_dataset.herb_ids[j] for j in top5_indices)

        # get true herbs for this syndrome from the multi-hot target
        true_herb_indices = true_herbs.nonzero(as_tuple=False).squeeze(1).tolist()
        true_herb_set = set(first_dataset.herb_ids[j] for j in true_herb_indices)

        # precision: of the 5 recommended herbs, how many were correct
        # recall: of all true herbs for this syndrome, how many did we find in our top 5
        true_pos = len(recommended_herb_set & true_herb_set)
        precision = true_pos / len(recommended_herb_set) if recommended_herb_set else 0
        recall = true_pos / len(true_herb_set) if true_herb_set else 0

        all_precision.append(precision)
        all_recall.append(recall)

print(f"\n── Overall Herb Recommendation Results ──")
print(f"Mean Precision@5: {np.mean(all_precision):.4f}")
print(f"Mean Recall@5:    {np.mean(all_recall):.4f}")


# Plot results (precision and recall histograms)
def plot_herb_recommendation_metrics(all_precision, all_recall):

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(12,5))

    # Precision histogram
    plt.subplot(1,2,1)
    plt.hist(all_precision, bins=20, alpha=0.7)
    plt.axvline(np.mean(all_precision), linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(all_precision):.2f}")
    plt.title("Precision@5 Distribution")
    plt.xlabel("Precision@5")
    plt.ylabel("Number of Syndromes")
    plt.legend()

    # Recall histogram
    plt.subplot(1,2,2)
    plt.hist(all_recall, bins=20, alpha=0.7)
    plt.axvline(np.mean(all_recall), linestyle="--", linewidth=2,
                label=f"Mean = {np.mean(all_recall):.2f}")
    plt.title("Recall@5 Distribution")
    plt.xlabel("Recall@5")
    plt.ylabel("Number of Syndromes")
    plt.legend()

    plt.tight_layout()

    # save figure instead of displaying
    plt.savefig("outputs/herb_recommendation_histograms.png", dpi=300)

    plt.close()

plot_herb_recommendation_metrics(all_precision, all_recall)
