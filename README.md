# Monkeypox-disease-diagnosis
A deep framework for Monkeypox disease diagnosis
# Dataset
The dataset used in this project is available at:

# Kaggle - Monkeypox Skin Image Dataset:
https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset
or
# Mendeley Data - Monkeypox Skin Image Dataset:
https://data.mendeley.com/datasets/r9bfpnvyxr/6

Please download the dataset from one of the above sources before running the project.
# Model Architecture
- The proposed model uses a feature extractor from a pre-trained Transformer model, followed by feature selection techniques such as:

1. Chi-Squared (Chi²)
2. Mutual Information
3. Recursive Feature Elimination (RFE)
4. Principal Component Analysis (PCA)

- The selected features are then classified using popular machine learning classifiers:

1. Random Forest
2. Support Vector Machine (SVM)
3. Naive Bayes
4. XGBoost

Feature scaling is applied using MinMaxScaler to normalize the data.

# How to Run
1. Install the required packages:
	* pip install -r requirements.txt
2. Download and prepare the dataset.
3. Run the (proposedmodel.py) script:
   * python proposedmodel.py
This script contains the proposed deep learning model for Monkeypox disease diagnosis.
# Requirements
The project uses the following main libraries:

* PyTorch
* Hugging Face Transformers
* scikit-learn
* XGBoost
* NumPy
* pandas
* Pillow
* matplotlib
* seaborn


