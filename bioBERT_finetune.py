import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, multilabel_confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from tqdm import tqdm
import pandas as pd
import seaborn as sns

class BioBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, multi_label_binarizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlb = multi_label_binarizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize input text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

def transform_data_from_json(json_path):
    """
    Transforms the JSON data into BioBERT-friendly format.
    """
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    formatted_data = []

    for entry in json_data:
        # Combine all relevant text fields
        antecedents = entry.get('Antecedents', [])
        symptoms = entry.get('Symptoms', [])
        
        # Create a comprehensive input text
        input_text = f"Antecedents: {' '.join(antecedents)}. Symptoms: {' '.join(symptoms)}"
        
        # Get diagnoses
        diagnoses = entry.get('Differential Diagnosis', [])
        
        formatted_data.append({
            "input_text": input_text,
            "label": diagnoses
        })

    return formatted_data

def prepare_datasets(train_data, val_data, test_data, all_labels, tokenizer, max_length):
    """
    Prepares datasets and multi-label binarizer.
    """
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])

    def encode_dataset(data):
        texts = [entry["input_text"] for entry in data]
        labels = [entry["label"] for entry in data]
        
        # Transform labels to binary matrix
        encoded_labels = mlb.transform(labels)

        print("Dataset Encoding:")
        print(f"Number of samples: {len(texts)}")
        print(f"Number of unique labels: {len(mlb.classes_)}\n")
        # print(f"Label classes: {mlb.classes_}")

        return texts, encoded_labels

    # Process datasets
    print("--- Train_Set ---")
    train_texts, train_labels = encode_dataset(train_data)
    print("--- Validation_Set ---")
    val_texts, val_labels = encode_dataset(val_data)
    test_texts = [entry["input_text"] for entry in test_data]

    # Create datasets
    train_dataset = BioBERTDataset(train_texts, train_labels, tokenizer, max_length, mlb)
    val_dataset = BioBERTDataset(val_texts, val_labels, tokenizer, max_length, mlb)

    return train_dataset, val_dataset, test_texts, mlb

def train_model(train_dataset, val_dataset, tokenizer, model_path="dmis-lab/biobert-v1.1", max_length=128, batch_size=32, epochs=50, lr=2e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Determine number of labels from the dataset
    num_labels = train_dataset.labels.shape[1]
    
    # Load pre-trained model with correct number of labels
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels, 
        problem_type="multi_label_classification"
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        "cosine", 
        optimizer=optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    patience = 3
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Option 2: iterate through all sequences in the batch (for debugging purposes)
                # input_ids_batch = batch["input_ids"]
                # for input_ids in input_ids_batch:
                #     print(tokenizer.decode(input_ids))

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

                # For multi-label, use sigmoid and threshold
                logits = outputs.logits # Do not do softmax
                val_pred_probs = torch.sigmoid(logits).cpu().numpy() # Now we apply sigmoid to each class
                val_predictions.extend(val_pred_probs)
                val_true_labels.extend(labels.cpu().numpy())

                # DEBUG: Only using softmax to get predicted class, just for printing.
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                # print("Probable Diagnosis:\n", predicted_classes)

        avg_val_loss = val_loss / len(val_loader)

        # Convert to numpy for metrics calculation
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)

        # Compute metrics
        val_accuracy = accuracy_score(val_true_labels.flatten(), (val_predictions > 0.5).astype(int).flatten())
        val_f1 = f1_score(val_true_labels, (val_predictions > 0.5).astype(int), average='samples')
        #val_accuracy = accuracy_score(val_true_labels, (val_predictions > 0.5).astype(int), multi_label=True)
        #val_f1 = f1_score(val_true_labels, (val_predictions > 0.5).astype(int), average='samples', multi_label=True)


        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save_pretrained("best_biobert_model")
            torch.save(model, "best_biobert_model/model.pt")  # Save the entire model
            tokenizer.save_pretrained("best_biobert_model")
            print("Best model saved!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model


def evaluate_model(test_texts, multi_label_binarizer, model_path="best_biobert_model", max_length=128, batch_size=32):
    """Evaluates the trained model and returns predictions, true labels, and metrics."""
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare the test dataset
    test_dataset = BioBERTDataset(
        test_texts,
        np.zeros((len(test_texts), len(multi_label_binarizer.classes_))),
        tokenizer,
        max_length,
        multi_label_binarizer
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize storage for predictions and probabilities
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Load batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get model outputs
            logits = model(input_ids, attention_mask=attention_mask).logits
            probabilities = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid for multi-label classification
            all_probabilities.extend(probabilities)

    # Convert probabilities to binary predictions using threshold
    all_probabilities = np.array(all_probabilities)
    all_predictions = (all_probabilities > 0.5).astype(int)

    # Placeholder for all_true_labels (for test comparison)
    all_true_labels = np.zeros_like(all_predictions)  # Replace with actual test labels for metric validation

    # Compute metrics
    test_accuracy = accuracy_score(all_true_labels.flatten(), all_predictions.flatten())
    test_f1 = f1_score(all_true_labels, all_predictions, average='samples', zero_division=0)
    hamming = hamming_loss(all_true_labels, all_predictions)

    metrics = {
        "accuracy": test_accuracy,
        "f1": test_f1,
        "hamming": hamming
    }

    return all_predictions, all_true_labels, metrics

def evaluate_model2(test_texts, multi_label_binarizer, model_path="/content/ML-Project---bioBERT/20k Samples/best_biobert_model", max_length=128, batch_size=32):
    """Evaluates the trained model and returns predictions, true labels, metrics, and top-5 probabilistic outputs."""
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare the test dataset
    test_dataset = BioBERTDataset(
        test_texts,
        np.zeros((len(test_texts), len(multi_label_binarizer.classes_))),
        tokenizer,
        max_length,
        multi_label_binarizer
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize storage for predictions and probabilities
    all_predictions = []
    all_probabilities = []
    top5_outputs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Load batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get model outputs
            logits = model(input_ids, attention_mask=attention_mask).logits
            probabilities = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid for multi-label classification
            all_probabilities.extend(probabilities)

    # Convert probabilities to binary predictions using threshold
    all_probabilities = np.array(all_probabilities)
    all_predictions = (all_probabilities > 0.5).astype(int)

    # Placeholder for all_true_labels (for test comparison)
    all_true_labels = np.zeros_like(all_predictions)  # Replace with actual test labels for metric validation

    # Extract top-5 probabilistic outputs
    label_classes = multi_label_binarizer.classes_
    for probs in all_probabilities:
        top5_indices = np.argsort(probs)[::-1][:5]  # Indices of top-5 probabilities in descending order
        top5_labels = [(label_classes[i], round(probs[i], 4)) for i in top5_indices]
        top5_outputs.append(top5_labels)

    # Compute metrics
    test_accuracy = accuracy_score(all_true_labels.flatten(), all_predictions.flatten())
    test_f1 = f1_score(all_true_labels, all_predictions, average='samples', zero_division=0)
    hamming = hamming_loss(all_true_labels, all_predictions)

    metrics = {
        "accuracy": test_accuracy,
        "f1": test_f1,
        "hamming": hamming
    }

    return all_predictions, all_true_labels, metrics, top5_outputs


# Main execution
def main():
    # File paths
    train_data_path = "dataset_processed/json/train_sample5000.json"
    val_data_path = "dataset_processed/json/val_sample2500.json"
    test_data_path = "dataset_processed/json/test_sample10000.json"

    # Full list of available labels
    with open('true_labels.json', 'r') as f:
        all_labels = json.load(f)

    # Transform data
    train_data = transform_data_from_json(train_data_path)
    val_data = transform_data_from_json(val_data_path)
    test_data = transform_data_from_json(test_data_path)

    # Prepare tokenizer
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    # Prepare datasets
    train_dataset, val_dataset, test_texts, label_encoder = prepare_datasets(
        train_data, val_data, test_data, all_labels, tokenizer, max_length=128
    )

    # Train model
    # model = train_model(train_dataset, val_dataset, tokenizer, epochs=50)

    # # Test model
    # all_predictions, all_true_labels, metrics = evaluate_model(test_texts, label_encoder)

    # if all_predictions is None:
    #     print("Evaluation failed. Check model path and files.")
    #     return

    # print("\n--- Evaluation Metrics ---")
    # print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    # # print(f"Test F1 Score (samples): {metrics['f1']:.4f}")
    # print(f"Test Hamming Loss: {metrics['hamming']:.4f}")

    # predictions_df = pd.DataFrame(all_predictions, columns=label_encoder.classes_)
    # true_labels_df = pd.DataFrame(all_true_labels, columns=label_encoder.classes_)
    # predictions_df.insert(0, 'Example ID', range(1, len(predictions_df) + 1))
    # true_labels_df.insert(0, 'Example ID', range(1, len(true_labels_df) + 1))

    # print("\n--- Predictions (First 20 Examples) ---")
    # print(predictions_df.head(20).to_string())
    # print("\n--- True Labels (First 20 Examples) ---")
    # print(true_labels_df.head(20).to_string())

    # print("\n--- Confusion Matrices (First 5 Labels) ---")
    # mcm = multilabel_confusion_matrix(all_true_labels, all_predictions)
    # for i in range(min(5, len(label_encoder.classes_))):
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=['Negative', 'Positive'],
    #                 yticklabels=['Negative', 'Positive'])
    #     plt.title(f'Confusion Matrix for: {label_encoder.classes_[i]}')
    #     plt.xlabel('Predicted Label')
    #     plt.ylabel('True Label')
    #     plt.show()

    # Test model
    all_predictions, all_true_labels, metrics, top5_outputs = evaluate_model2(
        test_texts=test_texts,
        multi_label_binarizer=label_encoder,
        model_path="best_biobert_model",
        max_length=128,
        batch_size=32
    )

    # Print metrics
    print("\n--- Evaluation Metrics ---")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Test Hamming Loss: {metrics['hamming']:.4f}")

    # Display top-5 predictions for the first 5 examples
    print("\n--- Top-5 Predictions (First 5 Examples) ---")
    for i, top5 in enumerate(top5_outputs[:5]):
        print(f"Example {i + 1}:")
        for label, prob in top5:
            print(f"  {label}: {prob}")
        print()

if __name__ == "__main__":
    main()