# ANN Classification for Date Fruit Recognition

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** using **PyTorch** to classify date fruits into seven different varieties based on their physical characteristics and extracted statistical features.

The model analyzes features such as fruit area, perimeter, major and minor axes, shape characteristics, color statistics, and texture-related measurements to predict the date fruit variety.

This is a supervised machine learning project that uses a neural network for **multi-class classification**.

## 🎯 Problem Statement

Different varieties of date fruits have distinct physical, geometrical, and statistical characteristics. Identifying these varieties manually can be time-consuming.

The objective of this project is to train an ANN model that automatically classifies date fruits into one of seven categories using numerical features extracted from the dataset.

## 📊 Dataset Information

The project uses the `DateFruit_dataset.csv` dataset.

- **Total records:** 898
- **Total columns:** 35
- **Input features:** 34
- **Target variable:** `Class`
- **Number of classes:** 7

### Date Fruit Classes

The model classifies date fruits into the following varieties:

1. BERHI
2. DEGLET
3. DOKOL
4. IRAQI
5. ROTANA
6. SAFAVI
7. SOGAY

### Features Used

The dataset contains several types of numerical features:

| Feature Category | Examples |
|---|---|
| Geometrical features | `AREA`, `PERIMETER`, `MAJOR_AXIS`, `MINOR_AXIS` |
| Shape characteristics | `ECCENTRICITY`, `SOLIDITY`, `ROUNDNESS`, `COMPACTNESS` |
| Shape factors | `SHAPEFACTOR_1`, `SHAPEFACTOR_2`, `SHAPEFACTOR_3`, `SHAPEFACTOR_4` |
| Color statistics | `MeanRR`, `MeanRG`, `MeanRB`, `StdDevRR`, `StdDevRG`, `StdDevRB` |
| Statistical features | Skewness, kurtosis, and entropy measurements for color channels |
| Wavelet-related features | `ALLdaub4RR`, `ALLdaub4RG`, `ALLdaub4RB` |

The `Class` column is the target variable that the ANN learns to predict.

## 🧠 Model Used: Artificial Neural Network (ANN)

The project implements a fully connected Artificial Neural Network using PyTorch.

### Model Architecture

- **Input Layer:** 34 input features
- **Hidden Layer 1:** 64 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation
- **Output Layer:** 7 neurons representing the seven date fruit classes

### How the Model Works

1. The model receives 34 numerical features describing a date fruit.
2. `StandardScaler` standardizes the input features before training.
3. The first hidden layer with 64 neurons learns relationships between the input features.
4. The second hidden layer with 64 neurons learns more complex patterns.
5. ReLU activation functions introduce nonlinearity into the network.
6. The output layer produces seven scores, one for each date fruit variety.
7. `CrossEntropyLoss` measures the classification error using the model's output scores and the correct class labels.
8. Backpropagation calculates gradients to improve the network's parameters.
9. The Adam optimizer updates the model weights during training.
10. The class with the highest output score is selected as the predicted date fruit variety.

## ⚙️ Technologies and Libraries

- **Python** — Programming language
- **PyTorch** — ANN implementation and training
- **Pandas** — Dataset loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Train-test splitting, feature scaling, and label encoding
- **Jupyter Notebook** — Development and experimentation

## 🔄 Project Workflow

1. Load `DateFruit_dataset.csv` using Pandas.
2. Inspect the dataset using `head()`, `tail()`, `describe()`, and `info()`.
3. Check for missing values and duplicate records.
4. Separate the 34 input features from the `Class` target column.
5. Convert the categorical target labels into numerical labels using `LabelEncoder`.
6. Split the dataset into training and testing sets using an 80:20 ratio with `random_state=42`.
7. Standardize the training and testing features using `StandardScaler`.
8. Convert the prepared features and labels into PyTorch tensors.
9. Create `TensorDataset` and `DataLoader` objects with a batch size of 32.
10. Define the ANN architecture with two hidden layers of 64 neurons each.
11. Train the model for 100 epochs using Adam and cross-entropy loss.
12. Evaluate the model on the test data and calculate classification accuracy.

## 📈 Model Performance

The notebook reports the following test accuracy:

| Evaluation Metric | Reported Value |
|---|---:|
| Test Accuracy | 93.89% |

The model correctly classified approximately 93.89% of the test samples in the reported evaluation.

This result indicates that the ANN learned useful patterns from the numerical features to distinguish between the seven date fruit varieties.

Actual performance may vary depending on data preprocessing, random initialization, and training settings.

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Susz-mishra/YOUR-REPOSITORY-NAME.git
```

Replace `YOUR-REPOSITORY-NAME` with the actual name of your GitHub repository.

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY-NAME
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy scikit-learn torch jupyter
```

### 4. Prepare the Dataset

Place `DateFruit_dataset.csv` in the location expected by the notebook.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `ANN_CLASSIFICATION.ipynb` and run the cells in order.

## 📁 Project Structure

```text
YOUR-REPOSITORY-NAME/
│
├── ANN_CLASSIFICATION.ipynb
├── DateFruit_dataset.csv
└── README.md
```

The dataset must be available for the notebook to execute successfully. The current notebook does not save the trained ANN model to a separate file.

## 🔍 Possible Improvements

- Fit `StandardScaler` on the training data only and use the same fitted scaler to transform the test data.
- Use a separate validation set to monitor model performance during training.
- Add a confusion matrix to identify which date fruit varieties are most frequently misclassified.
- Calculate precision, recall, and F1-score for each class.
- Experiment with different numbers of hidden layers and neurons.
- Compare the ANN with algorithms such as Random Forest, Support Vector Machine, and K-Nearest Neighbors.
- Save the trained model and fitted scaler for reuse.
- Build a simple web application where users can enter fruit measurements and obtain a predicted class.

## ⚠️ Limitations

- The model depends on the quality and representativeness of the dataset.
- The dataset contains 898 records, so performance on larger or different datasets may vary.
- The reported accuracy alone does not reveal performance for each individual class.
- The notebook currently fits a separate scaler on the test data. For a more reliable evaluation, fit the scaler only on the training data and transform the test data with that same scaler.
- The project classifies numerical feature records; it does not directly classify a raw fruit photograph.

## 👨‍💻 Author

**Anuj Kumar Mishra**

GitHub: [Susz-mishra](https://github.com/Susz-mishra)

---

⭐ If you find this project useful, consider starring the repository.
