import numpy as np

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize
# all metrics: https://scikit-learn.org/stable/api/sklearn.metrics.html
from sklearn.metrics import accuracy_score, auc, classification_report, roc_auc_score, f1_score, confusion_matrix, roc_curve,precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

data = load_dataset("gretelai/symptom_to_diagnosis") #hugging face dataset
train_symptoms, train_label = list(data["train"]["input_text"]), list(data["train"]["output_text"])
test_symptoms, test_label = list(data["test"]["input_text"]), list(data["test"]["output_text"])
train_df = pd.DataFrame(data["train"])
test_df = pd.DataFrame(data["test"])

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# parameter documentation https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# all parameters https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


crossval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_model(model_name, model):

    cv_accuracy = cross_val_score(model, train_symptoms, train_label, cv=crossval, scoring="accuracy")
    cv_f1 = cross_val_score(model, train_symptoms, train_label, cv=crossval, scoring="f1_weighted")

    print(model_name)
    print("CV accuracy at every fold:", cv_accuracy.round(3))
    print("CV accuracy:", round(cv_accuracy.mean(), 3), "std", round(cv_accuracy.std(), 3))
    print("CV f1 score:", round(cv_f1.mean(), 3), "std", round(cv_f1.std(), 3))
    model.fit(train_symptoms, train_label)
    clf = model.predict(test_symptoms)

    classes = model.classes_
    binary_matrix = label_binarize(test_label, classes=classes)
    proba = model.predict_proba(test_symptoms)

    auc_score = roc_auc_score(binary_matrix, proba, average='weighted', multi_class='ovr')
    f1 = f1_score(test_label, clf, average='weighted')
    precision = precision_score(test_label, clf, average='weighted', zero_division=0)
    recall = recall_score(test_label, clf, average='weighted')

    print("accuracy:", accuracy_score(test_label, clf))
    print('f1 score:', f1, "AUC score", auc_score)
    print('Overall precision:', precision)
    print('Overall recall:', recall)

    # eval overfitting/under
    train_pred = model.predict(train_symptoms)
    train_accuracy = accuracy_score(train_label, train_pred)
    test_accuracy = accuracy_score(test_label, clf)

    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
    print("Gap", train_accuracy - test_accuracy)

    print(classification_report(test_label, clf))
    #print('confusion matrix', confusion_matrix(test_label, clf))

    return {"model": model_name,
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score}

log_reg = make_pipeline(
    TfidfVectorizer(max_features=500),
    OneVsRestClassifier(LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000))
)

naive_bayes = make_pipeline(
    TfidfVectorizer(max_features=500),
    MultinomialNB()
)

svm = make_pipeline(
    TfidfVectorizer(max_features=500),
    SVC(kernel="linear", probability=True, random_state=42)
)

decision_tree = make_pipeline(
    TfidfVectorizer(max_features=500),
    DecisionTreeClassifier(random_state=42)
)

random_forest = make_pipeline(
    TfidfVectorizer(max_features=500),
    RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
)

results = [
    evaluate_model("Logistic Regression", log_reg),
    evaluate_model("Naive Bayes", naive_bayes),
    evaluate_model("Linear SVM", svm),
    evaluate_model("Decision Tree", decision_tree),
    evaluate_model("Random Forest", random_forest),
]

results_df = pd.DataFrame(results)
print(results_df.round(3))


