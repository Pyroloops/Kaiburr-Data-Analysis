# Task 5: Consumer Complaint Text Classification Project

This project implements a complete Machine Learning workflow to classify consumer complaints from the Consumer Financial Protection Bureau (CFPB) based on the text narrative provided by the consumer.

The project follows the steps outlined in the assessment guide, covering data cleaning, feature engineering using TF-IDF, training and comparison of three classification models, and final performance evaluation.

## 1. Project Goal

The primary goal is to build a robust classifier that can accurately categorize consumer complaints into one of four target product categories:

* 1. Credit reporting, credit repair services, or other personal consumer reports

* 2. Debt collection

* 3. Consumer Loan

* 4. Mortgage

## 2. Setup and Execution

## 3. Prerequisites

To run the script successfully, you must have Python installed, along with the required libraries.

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

## 4. Data Source

The script loads the data directly from the official CFPB complaint database URL:
https://files.consumerfinance.gov/ccdb/complaints.csv.zip

## 5. Script Execution

The entire workflow is contained in the text_classification_task.py file. Run the script using the Python interpreter:

```bash
python text_classification_task.py
```

## 6. Machine Learning Workflow

The pipeline is structured into four main phases:

### 1. Phase I: Data Preparation (Section 6.2)

* 2. Loading and Filtering: The script loads the data, selects the Product and Consumer complaint narrative columns, and removes any rows where the narrative is missing.

* 3. Category Selection: The dataset is filtered to include only the four target categories specified in the assessment.

* 4. Text Preprocessing: A custom function converts text to lowercase, removes punctuation and numbers, and eliminates common English stopwords (using NLTK) to prepare the narratives for vectorization.

### 2. Phase II: Feature Engineering (Section 6.3)

* 1. Technique: The script uses TF-IDF (Term Frequency-Inverse Document Frequency) via scikit-learn's TfidfVectorizer.

* 2. Purpose: This process converts the clean text narratives into a numerical feature matrix, where each column represents a word or n-gram, and the values represent its importance to that specific document.

### 3. Phase III: Model Training and Comparison (Section 6.4)

* 1. Data Split: The data is split into 75% for training and 25% for testing using train_test_split with stratification.

* 2. Model Pipelines: A scikit-learn Pipeline is used for each model, chaining the TfidfVectorizer and the classifier. This prevents data leakage and simplifies the workflow.
 
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/c898223f-150b-43ae-a3db-25e3d7fb4df1" />

* 3. Models Trained:

MultinomialNB (Multinomial Naive Bayes)

LogisticRegression

LinearSVC (Linear Support Vector Machine)

### 4. Phase IV: Detailed Evaluation 

### Performance Comparison: 

| Model               | Accuracy  | Precision | Recall   | F1-Score  |
|--------------------|-----------|-----------|---------|-----------|
| MultinomialNB       | 0.862349  | 0.707728  | 0.721099 | 0.708091 |
| LogisticRegression  | 0.908779  | 0.848243  | 0.768609 | 0.796689 |
| LinearSVC           | 0.909467  | 0.869662  | 0.751412 | 0.783990 |

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/ec86951b-f348-4753-a7c8-3fc3791fddbf" />

### Confusion Matrix:

<img width="1170" height="989" alt="image" src="https://github.com/user-attachments/assets/a4890610-a747-44cc-9d55-af839b1fb5ff" />

## 7. Model Saveing

The final best-performing pipeline (TfidfVectorizer + LinearSVC) along with the other two models are saved using the pickle library, allowing the models to be loaded and used for new predictions without having to retrain it.
