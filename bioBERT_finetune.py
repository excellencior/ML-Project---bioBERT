import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from tqdm import tqdm

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
                print("Most Probable Diagnosis:\n", predicted_classes)

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
            tokenizer.save_pretrained("best_biobert_model")
            print("Best model saved!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model

def evaluate_model(test_texts, multi_label_binarizer, model_path="best_biobert_model", max_length=128, batch_size=32):
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Prepare test dataset
    test_dataset = BioBERTDataset(
        test_texts, 
        np.zeros((len(test_texts), len(multi_label_binarizer.classes_))), 
        tokenizer, 
        max_length, 
        multi_label_binarizer
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions and probabilities
    all_predictions = []
    all_probabilities = []
    all_true_labels=[]

    with torch.no_grad():
         for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device) #Get true labels, so they can be used in metrics
            # Get logits and apply sigmoid
            logits = model(input_ids, attention_mask=attention_mask).logits
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            
            # Convert to binary predictions with different thresholds
            binary_predictions_05 = (probabilities > 0.5).astype(float)

            all_predictions.extend(binary_predictions_05)
            all_probabilities.extend(probabilities)
            all_true_labels.extend(labels.cpu().numpy())

    # Convert to NumPy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_true_labels = np.array(all_true_labels)


    # Prepare result with labels, binary predictions, and probabilities
    results = []
    for pred, prob, label in zip(all_predictions, all_probabilities,all_true_labels):
        # Get labels for this prediction
        labels_05 = multi_label_binarizer.classes_[pred.astype(bool)]
        
        # Get corresponding probabilities
        label_probs = {label: prob[idx] for idx, label in enumerate(multi_label_binarizer.classes_) if pred[idx] > 0}
        
        results.append({
            "labels_05": list(labels_05),
            "probabilities": label_probs,
            "true_labels": multi_label_binarizer.classes_[label.astype(bool)]
        })
    
    # Compute metrics
    test_accuracy = accuracy_score(all_true_labels.flatten(), all_predictions.flatten())
    test_f1 = f1_score(all_true_labels, all_predictions, average='samples')
    hamming = hamming_loss(all_true_labels, all_predictions)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Hamming Loss: {hamming:.4f}")

    return results

# Main execution
def main():
    # File paths
    train_data_path = "dataset_processed/json/train_sample1000.json"
    val_data_path = "dataset_processed/json/val_sample250.json"
    test_data_path = "dataset_processed/json/test_sample1000.json"

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
    model = train_model(train_dataset, val_dataset, tokenizer, epochs=80)

    # Evaluate model
    predicted_labels = evaluate_model(test_texts, label_encoder)

    # Print results
    print("\nTest Predictions (first 10 examples):")
    for i, result in enumerate(predicted_labels[:10]):
        print(f"Example {i + 1}:")
        print("Labels (threshold 0.5):", result["labels_05"])
        print("Probabilities:", result["probabilities"])
        print("---")

if __name__ == "__main__":
    main()