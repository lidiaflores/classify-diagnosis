# classify-diagnosis
a repo for supervised classification of medical diagnosis

compares a baseline TF-IDF approach against sentence embeddings. 

project uses the HuggingFace dataset [gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)

This is a learning project and still a work in progress, it should not be used for medical decisions. 

## To re-run the analysis 
1. run `baseline/compare_models.py` for TF-IDF approach
2. run `embedding/compare_embed.py` for embedding approach (in progress)
