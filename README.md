>Dialogue Act Classification Using DistilBERT
Project Overview
This project implements dialogue act classification using the DistilBERT transformer model. The aim is to classify dialogues into predefined categories (e.g., inform, question, directive, etc.). The project utilizes fine-tuning of DistilBERT and incorporates preprocessing, tokenization, and training with PyTorch and Hugging Face's Trainer API.

Directory Structure
.
├── train.csv            # Training dataset
├── test.csv             # Test dataset
├── validation.csv       # Validation dataset
├── results/             # Output directory for model checkpoints and logs
├── dialogue_model_hmtl/ # Directory to save the trained model and tokenizer
├── main.py              # Main script for training and evaluation
└── README.md            # Project documentation

>>Key Features
Preprocessing:
Converts dialogue text to lowercase, removes extra spaces, and handles malformed annotations in the act labels.

Tokenization:
Uses DistilBERT Tokenizer to encode dialogue texts into input IDs and attention masks.

Model:
DistilBERTForSequenceClassification fine-tuned for multi-class classification with 5 dialogue act categories.

Trainer API:
Simplifies training with features like early stopping, checkpointing, and evaluation on the validation set.

Evaluation:
Reports validation accuracy and loss, and saves the best-performing model.

Usage Instructions
1. Installation
Install the required libraries:
pip install pandas transformers sklearn torch numpy

2. Preprocess the Data
Ensure the datasets (train.csv, test.csv, validation.csv) are present in the project directory. Preprocessing is applied to clean and tokenize the data during execution.

3. Train the Model
Run the main script to train the model:

The training arguments, such as the number of epochs and batch size, can be adjusted in the TrainingArguments section.

4. Evaluate the Model
The script will automatically evaluate the model after each epoch on the validation dataset and save the best model based on validation loss.

5. Save and Load the Model
The trained model and tokenizer are saved to ./dialogue_model_hmtl/. Load the saved model for inference:

Results:
Validation Accuracy: Achieved 71% accuracy after fine-tuning.

>>Future Work
Explore advanced transformer models like BERT or RoBERTa.
Experiment with hierarchical classification for improved performance.
Address imbalances in the dataset for enhanced prediction reliability.
