import optuna
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PART 1: LOADING AND PREPARING THE DATASETS

# Load the datasets
olid_train = pd.read_csv('/content/olid-train-small.csv')
olid_test = pd.read_csv('/content/olid-test.csv')
hasoc_train = pd.read_csv('/content/hasoc-train.csv')

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Turn the train and test datasets into a pandas dataframe
olid_train_data = pd.DataFrame({
    'text': olid_train['text'],
    'labels': olid_train['labels']
})

olid_test_data = pd.DataFrame({
    'text': olid_test['text'],
    'labels': olid_test['labels']
})

hasoc_train_data = pd.DataFrame({
    'text': hasoc_train_set['text'],
    'labels': hasoc_train_set['labels']
})

# PART 2: TUNING

# split the training dataset into a training and validation set for tuning
train_df, eval_df = train_test_split(olid_train_data, test_size=0.2) # for the cross-domain exp. use 'hasoc_train_data'
                                                                     # for the in-domain exp. use 'olid_train_data'

# Create a function for Optuna to tune the hyperparameters
def objective(trial):
    # For each hyperparameter use various numbers
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 5)
    train_batch_size = trial.suggest_categorical('train_batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    
    # Create the model parameters with the hyperparameters suggested above
    model_args = {
        'num_train_epochs': num_train_epochs,
        'train_batch_size': train_batch_size,
        'learning_rate': learning_rate,
        'overwrite_output_dir': True,
        'eval_batch_size': 64,  # keeping constant
        'silent': True  # suppress the output for a faster tuning process
    }
    
    # Create a BERT model with the model arguments suggested in the 'models_arg'
    model = ClassificationModel('bert', 'bert-base-cased', args=model_args, use_cuda=True)
    
    # Train the BERT model on the training set
    model.train_model(train_df)
    
    # Evaluate the performance of the BERT model on the evaluation set
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=accuracy_score) # metric used is accuracy
    
    # Return accuracy which will be used to optimize the hyperparameters
    return result['acc']

# Initialize the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # take 10 trials

# Print the best parameters
print("Best hyperparameters found: ", study.best_params)

# PART 3: TRAINING THE MODEL USING THE OPTIMIZED HYPERPARAMETERS

# Obtain the best parameters from the tuning process
best_params = study.best_params

# initialize the model arguments using the best parameters
model_args = {
    'num_train_epochs': best_params['num_train_epochs'],
    'train_batch_size': best_params['train_batch_size'],
    'learning_rate': best_params['learning_rate'],
    'overwrite_output_dir': True,
    'eval_batch_size': 64,
    'silent': False 
}

# Create the BERT model for classification
model_bert = ClassificationModel(
    "bert", "bert-base-cased", 
    args=model_args, 
    use_cuda=True
)

# Train the model on the training dataset (in-domain or cross-domain)
model_bert.train_model(olid_train_data) # for the cross-domain exp. use 'hasoc_train_data'
                                        # for the in-domain exp. use 'olid_train_data'


# Evaluate the model on the test set which is 'olid_test_data' for both in-domain and cross-domain
result, model_outputs, wrong_predictions = model_bert.eval_model(olid_test_data)

# Extract the hate labels from the test set
true_labels = olid_test_data['labels'].values

# Convert the output labels from BERT into the right format
predicted_labels = np.argmax(model_outputs, axis=1)

# PART 4: CALCULATE THE EVALUATION METRICS

# Obtain the macro-averaged precision, recall, and F1-score
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')

# Obtain the per-class precision, recall, and F1-score
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

# Print model results
print(f"Macro-averaged Precision: {precision_macro:.4f}")
print(f"Macro-averaged Recall: {recall_macro:.4f}")
print(f"Macro-averaged F1-score: {f1_macro:.4f}")

# Print class results
print("\nPer-Class Precision, Recall, and F1-score:")
for i, (p, r, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
    print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f1:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Create the confusion matrix plot
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted: Non-Offensive', 'Predicted: Offensive'],
            yticklabels=['Actual: Non-Offensive', 'Actual: Offensive'])

# Add labels and title
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('In-Domain Confusion Matrix') #change title to cross-domain confusion matrix

# Create the figure
plt.show()




