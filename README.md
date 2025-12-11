# DistilBERT
Final Project for Machine Learning Programming. 
We use the DistilBERT model from HuggingFace: https://github.com/huggingface/transformers/tree/main/src/transformers/models/distilbert
We test the model on AG News dataset and change training parameters to improve performance. 

data_loader.py        # Load the AG News dataset \\
train_baseline.py     # Train DistilBERT baseline \\
train_contribution.py # Train DistilBERT with modified parameters \\
evaluate.py           # Evaluate the models \\
mlp_baseline.py       # Simple MLP model for comparison
