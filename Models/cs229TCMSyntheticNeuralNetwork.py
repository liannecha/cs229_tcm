"""
Owner: Ethan

Same as cs229TCMNeuralNetworkSyndrome, but trained on synthetic patient data.
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

class SyntheticTCMDataset(Dataset):
    def __init__(self, synthetic_X_file, synthetic_y_file, concept_file):
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.concept_targets[idx], self.y_syndrome[idx]

class TCMNet(nn.Module):
    def __init__(self, num_symptoms=1717, num_concepts=14, num_syndromes=233):
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

        return concept_preds, syndrome_preds

# now training loop
def train_model(model, dataloader, epochs=100, class_weights=None, lambda_concept=1.0, lambda_syndrome=1.0):
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

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'total': [], 'concept': [], 'syndrome': []}

    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_syndrome = 0.0
        epoch_concept = 0.0

        for symptoms, true_concepts, true_syndromes in dataloader:
            optimizer.zero_grad() # zero the parameter gradients

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

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {history['total'][-1]:.4f}")
        
    return history



# Now we initialise and train the model
synthetic_x_file = "Synthetic_Patient_Symptoms.csv"
synthetic_y_file = "Synthetic_Patient_Labels.csv"
concept_file = "Syndrome_Concept_Targets.csv"

# Create dataset and dataloader
first_dataset = SyntheticTCMDataset(synthetic_x_file, synthetic_y_file, concept_file)
dataset_size = len(first_dataset)

# split into train and validation sets
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

print(f"Dataset loaded with {len(np.unique(first_dataset.y_syndrome))} syndromes and {first_dataset.X.shape[1]} symptoms.")

# create the dataloaders
train_dataset, val_dataset = random_split(first_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# initialise the model
model = TCMNet(num_symptoms=first_dataset.X.shape[1], num_concepts=14, num_syndromes=len(np.unique(first_dataset.y_syndrome)))

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
    lambda_syndrome = 1.0
)

def evaluate_model(trained_model, dataloader):
    trained_model.eval() # set model to evaluation mode

    correct_syndromes = 0
    total_samples = 0
    total_concept_error = 0.0

    criterion_concept = nn.MSELoss()

    with torch.no_grad(): # turn off gradients for evaluation
        for symptoms, true_concepts, true_syndromes in dataloader:

            # forward pass
            pred_concepts, pred_syndromes = trained_model(symptoms.float())

            # evaluate syndrome predictions
            _, predicted_syndromes = torch.max(pred_syndromes, 1)
            correct_syndromes += (predicted_syndromes == true_syndromes).sum().item()
            total_samples += true_syndromes.size(0)

            # evaluate concept predictions
            concept_error = criterion_concept(pred_concepts, true_concepts)
            total_concept_error += concept_error.item()

    syndrome_accuracy = (correct_syndromes / total_samples) * 100
    avg_concept_error = total_concept_error / len(dataloader)

    print("Evaluation Results:")
    print(f"Syndrome Classification Accuracy: {syndrome_accuracy:.2f}%")
    print(f"Average Concept Prediction MSE (Lower is better): {avg_concept_error:.4f}")

evaluate_model(model, val_loader) # do it on the validation set now

def plot_loss_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['total'], label='Total Loss', color='black', linewidth=2)
    plt.plot(history['concept'], label='Concept Loss (MSE)', linestyle='--')
    plt.plot(history['syndrome'], label='Syndrome Loss (Cross Entropy)', linestyle='-.')
    
    plt.title('Training Loss Curve', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.show()

plot_loss_curve(loss_history)

def plot_patient_profile(model, dataset, index=0):
    model.eval() # evaluation mode

    symptoms, true_concepts, true_syndromes = dataset[index]
    syndrome_name = dataset.syndrome_names[index]

    with torch.no_grad():
        pred_concepts, pred_syndromes = model(symptoms.unsqueeze(0).float())

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

    with torch.no_grad(): # turn off gradients for evaluation
        for symptoms, true_concepts, true_syndromes in dataloader:

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
