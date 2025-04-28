import torch
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from PIL import Image
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize

# Dataset Paths
dataset_dir = '..//Monkeypox_Skin_Image_Dataset/dataset'

# Get class names and paths for each image
class_names = os.listdir(dataset_dir)
image_paths, labels = [], []

for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(class_name)

# Encode labels
label_to_index = {name: idx for idx, name in enumerate(class_names)}
labels_encoded = np.array([label_to_index[label] for label in labels])

# Feature Extractor and Model (Swin Transformer)
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")
model = AutoModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
num_classes = len(class_names)

# Modify model to add classification head
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dataset Class
class SkinImageDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}  # Remove batch dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs, label

# Fine-tune the Swin Transformer
def fine_tune_swin(model, train_dataloader, epochs=3, lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_dataloader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.last_hidden_state.mean(dim=1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

# Extract features using the fine-tuned model
def extract_features_swin(model, dataloader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            features.append(logits)
            labels.extend(batch_labels.numpy())

    features = np.vstack(features)
    labels = np.array(labels)
    return features, labels

# Feature Selection Options
def feature_selection(method, train_features, train_labels, val_features, k=500):
    if method == "SelectKBest":
        selector = SelectKBest(chi2, k=k)
    elif method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == "RFE":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=k, step=0.1)
    elif method == "PCA":
        selector = PCA(n_components=k)
        train_features = selector.fit_transform(train_features)
        val_features = selector.transform(val_features)
        return train_features, val_features
    elif method == "VarianceThreshold":
        selector = VarianceThreshold(threshold=0.01)
    else:
        raise ValueError("Invalid feature selection method specified.")
    
    train_features_selected = selector.fit_transform(train_features, train_labels)
    val_features_selected = selector.transform(val_features)
    return train_features_selected, val_features_selected

# Cross-Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
fold = 1

# Classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "SVM": SVC(C=10, gamma=0.01, kernel='rbf', probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42)
}
from sklearn.metrics import roc_curve, auc  # Ensure correct import of `auc`

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming `labels_encoded` contains integer labels (0, 1, 2, ...)
n_classes = len(class_names)  # Number of classes

for train_idx, val_idx in cv.split(image_paths, labels_encoded):
    print(f"\n===== Fold {fold} =====")
    train_paths = [image_paths[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    train_labels = labels_encoded[train_idx]
    val_labels = labels_encoded[val_idx]

    # Binarize labels for multi-class ROC/AUC
    train_labels_bin = label_binarize(train_labels, classes=range(n_classes))
    val_labels_bin = label_binarize(val_labels, classes=range(n_classes))

    # Create Dataloaders
    train_dataset = SkinImageDataset(train_paths, train_labels, feature_extractor)
    val_dataset = SkinImageDataset(val_paths, val_labels, feature_extractor)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # Fine-Tune the Model
    fine_tune_swin(model, train_dataloader, epochs=3)

    # Extract Features
    train_features, train_labels = extract_features_swin(model, train_dataloader)
    val_features, val_labels = extract_features_swin(model, val_dataloader)

    # Normalize Features
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Feature Selection
    for method in ["SelectKBest", "mutual_info", "RFE", "PCA"]:
        print(f"\nUsing Feature Selection Method: {method}")
        train_features_selected, val_features_selected = feature_selection(
            method, train_features, train_labels, val_features, k=500
        )

        # Train and Evaluate Classifiers
        for clf_name, clf in classifiers.items():
            print(f"\nTraining {clf_name}...")
            clf.fit(train_features_selected, train_labels)
            val_preds = clf.predict(val_features_selected)
            val_probs = clf.predict_proba(val_features_selected) if hasattr(clf, "predict_proba") else None

            # Metrics
            accuracy = accuracy_score(val_labels, val_preds)
            auc_score = roc_auc_score(val_labels_bin, val_probs, multi_class="ovr") if val_probs is not None else None
            report = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
            conf_matrix = confusion_matrix(val_labels, val_preds)

            print(f"{clf_name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}" if auc_score else f"{clf_name} - Accuracy: {accuracy:.4f}")

            # Save Metrics
            results.append({
                "Fold": fold,
                "Feature_Selection": method,
                "Classifier": clf_name,
                "Accuracy": accuracy,
                "AUC": auc_score,
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-Score": report["weighted avg"]["f1-score"]
            })

            # Plot Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"Confusion Matrix: {method} + {clf_name}", fontsize=16)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("True Label", fontsize=12)
            plt.tight_layout()
            plt.show()

                       # Plot ROC Curve for Multi-Class
            if val_probs is not None:
                plt.figure(figsize=(8, 6))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(val_labels_bin[:, i], val_probs[:, i])
                    roc_auc = auc(fpr, tpr)*100
                    
                    # Scale to 0-100 range
                    fpr = fpr * 100
                    tpr = tpr * 100
                    
                    plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

                plt.plot([0, 100], [0, 100], color='gray', linestyle='--')  # Diagonal line for random performance
                plt.xlim([0.0, 105.0])
                plt.ylim([0.0, 105.0])
                plt.xlabel('False Positive Rate (%)', fontsize=12)
                plt.ylabel('True Positive Rate (%)', fontsize=12)
                plt.title(f'ROC Curve: {method} + {clf_name}', fontsize=16)
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.show()


    fold += 1


# Save Results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("cross_validation_results.csv", index=False)

# Calculate Mean Metrics for Each Feature Selection Method and Classifier
mean_results = results_df.groupby(["Feature_Selection", "Classifier"]).mean().reset_index()
mean_results.to_csv("cross_validation_mean_results.csv", index=False)

print("\nAll results saved to cross_validation_results.csv.")
print("\nMean results saved to cross_validation_mean_results.csv.")

# Load Mean Results
mean_results = pd.read_csv("cross_validation_mean_results.csv")

# Plot Graphs Based on Mean Results
metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1-Score"]
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=mean_results,
        x="Feature_Selection",
        y=metric,
        hue="Classifier",
        palette="viridis"
    )
    plt.title(f"Mean {metric} by Feature Selection Method and Classifier", fontsize=16)
    plt.xlabel("Feature Selection Method", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=30)
    plt.legend(title="Classifier", fontsize=10)
    plt.tight_layout()
    plt.show()

# Heatmap for Metrics
metrics_to_plot = ["Accuracy", "AUC", "Precision", "Recall", "F1-Score"]
for metric in metrics_to_plot:
    heatmap_data = mean_results.pivot("Feature_Selection", "Classifier", metric)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title(f"Heatmap of {metric} Across Feature Selection Methods and Classifiers", fontsize=16)
    plt.xlabel("Classifier", fontsize=12)
    plt.ylabel("Feature Selection Method", fontsize=12)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Confusion Matrix Based on Mean
# Calculate the mean confusion matrix (mocked example for now)
mean_conf_matrix = np.array([[50, 10], [7, 85]])  # Replace with your actual average confusion matrix

plt.figure(figsize=(8, 6))
sns.heatmap(mean_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Mean Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(mean_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Confusion Matrix: {method} + {clf_name}", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.show()

# Radar Chart for Feature Selection Method
from math import pi

def plot_radar_chart(selection_method):
    subset = mean_results[mean_results["Feature_Selection"] == selection_method]
    categories = ["Accuracy", "AUC", "Precision", "Recall", "F1-Score"]
    num_vars = len(categories)

    # Radar chart setup
    plt.figure(figsize=(8, 8))
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    for classifier in subset["Classifier"].unique():
        values = subset[subset["Classifier"] == classifier][categories].values.flatten().tolist()
        values += values[:1]
        plt.polar(angles, values, label=classifier)

    plt.title(f"Radar Chart for Feature Selection Method: {selection_method}", fontsize=16)
    plt.xticks(angles[:-1], categories, fontsize=12)
    plt.fill(angles, values, alpha=0.1)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()

# Generate Radar Charts for Each Feature Selection Method
for method in mean_results["Feature_Selection"].unique():
    plot_radar_chart(method)

# Pairplot for Metrics Comparison
sns.pairplot(mean_results, vars=metrics, hue="Classifier", palette="Set2", diag_kind="kde")
plt.suptitle("Pairwise Comparison of Metrics by Classifier", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Save all plots and metrics for further analysis
mean_results.to_csv("final_mean_metrics.csv", index=False)
print("Final metrics and plots have been saved.")

# Identify the best classifier and feature selection method based on Accuracy from the mean results
best_classifier = mean_results.loc[mean_results['Accuracy'].idxmax()]
best_feature_selection = mean_results.groupby("Feature_Selection")["Accuracy"].mean().idxmax()

# Final graph for the best classifier
best_classifier_data = mean_results[mean_results["Classifier"] == best_classifier["Classifier"]]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=best_classifier_data,
    x="Feature_Selection",
    y="Accuracy",
    palette="viridis"
)
plt.title(f"Accuracy for Best Classifier: {best_classifier['Classifier']}", fontsize=16)
plt.xlabel("Feature Selection Method", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Final graph for the best feature selection method
best_feature_data = mean_results[mean_results["Feature_Selection"] == best_feature_selection]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=best_feature_data,
    x="Classifier",
    y="Accuracy",
    palette="mako"
)
plt.title(f"Accuracy for Best Feature Selection Method: {best_feature_selection}", fontsize=16)
plt.xlabel("Classifier", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Print the best classifier and best feature selection method for reference
print("Best Classifier:", best_classifier["Classifier"])
print("Best Feature Selection Method:", best_feature_selection)

# After training and evaluation for each fold and classifier

for clf_name, clf in classifiers.items():
    # Assuming val_preds and val_labels are the predictions and true labels for the validation set
    val_preds = clf.predict(val_features_selected)
    val_probs = clf.predict_proba(val_features_selected) if hasattr(clf, "predict_proba") else None

    # Confusion Matrix
    conf_matrix = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for {clf_name}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    # ROC Curve (for multi-class classification)
    if val_probs is not None:
        n_classes = len(np.unique(val_labels))
        fpr, tpr, roc_auc = {}, {}, {}
        
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(val_labels == i, val_probs[:, i])
            roc_auc[i] = roc_auc_score(val_labels == i, val_probs[:, i])
            
            # Plot ROC curve for each class
            plt.figure(figsize=(8, 6))
            plt.plot(fpr[i], tpr[i], color='darkorange', lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f"ROC Curve for {clf_name}", fontsize=16)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"{clf_name} does not support probability predictions for ROC curve.")

