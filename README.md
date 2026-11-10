# classify-diagnosis
a repo for supervised classification of medical diagnosis

project uses the HuggingFace dataset [gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) to classify 22 different diseases based on a synthetic dataset of patient's describing their symptoms.  

This project compares a baseline TF-IDF approach against an embedding approach with hyperparameter optimization through cross-validation. 

## models
Both approaches are tested with logistic regression (one v rest), linear SVM,
decision tree, and random forest, using 5-fold cross validation on the train split. multinomial NB is only used in TF-IDF approach. 

The embedding approach used GridSearchCV for hyperparameter tuning, with settings selected through 5-fold cross validation, then each tuned model was evaluated once on the test split. Calibration scores, brier and log loss, are also reported. Stratified bootstrap simulations (n=1,000) of the test set were conducted, and 95% confidence intervals are reported. 

## To re-run the analysis 
1. run `baseline/compare_models.py` for TF-IDF approach
2. run `embedding/embedding_models.ipynb` for embedding approach 

## limitations 
- Huggingface dataset is synthetic, may not represent real text on patient symptoms 
- This is a learning project, it should not be used for medical decisions. 
