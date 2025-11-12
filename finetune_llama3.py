"""
Fine-tuning LLaMA-3 for Fake News Classification

This script fine-tunes the LLaMA-3 model for binary classification
of fake vs. real news articles using LoRA (Low-Rank Adaptation).
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix, classification_report, 
    balanced_accuracy_score, accuracy_score
)

from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

class ModelConfig:
    """Configuration class for model parameters."""
    
    @staticmethod
    def get_quantization_config():
        """Get quantization configuration for efficient training."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    @staticmethod
    def get_lora_config():
        """Get LoRA configuration for parameter-efficient fine-tuning."""
        return LoraConfig(
            r=16,
            lora_alpha=8,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            bias='none',
            task_type='SEQ_CLS'
        )

class DatasetLoader:
    @staticmethod
    def load_datasets():
        """Load train, validation, and test datasets."""
        try:
            df_train = pd.read_csv('df_train.csv')
            df_val = pd.read_csv('df_val.csv')
            df_test = pd.read_csv('df_test.csv')
            
            logger.info(f"Loaded datasets - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
            return df_train, df_val, df_test
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise
    
    @staticmethod
    def create_huggingface_datasets(df_train, df_val, df_test):
        """Convert pandas DataFrames to HuggingFace Dataset objects."""
        dataset = DatasetDict({
            'train': Dataset.from_pandas(df_train),
            'val': Dataset.from_pandas(df_val),
            'test': Dataset.from_pandas(df_test)
        })
        return dataset
    
    @staticmethod
    def calculate_class_weights(df_train):
        """Calculate class weights for balanced training."""
        class_weights = (1 / df_train.label.value_counts(normalize=True).sort_index()).tolist()
        class_weights = torch.tensor(class_weights)
        class_weights = class_weights / class_weights.sum()
        logger.info(f"Class weights: {class_weights}")
        return class_weights

class ModelTrainer:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def setup_model(self):
        """Initialize and configure the model for training."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load model with quantization
        quantization_config = ModelConfig.get_quantization_config()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            num_labels=2,
            trust_remote_code=True
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        lora_config = ModelConfig.get_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("Model setup complete")
    
    def setup_tokenizer(self):
        """Initialize and configure the tokenizer."""
        logger.info("Setting up tokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model for tokenizer
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
    
    def preprocess_datasets(self, dataset):
        """Tokenize and preprocess the datasets."""
        logger.info("Preprocessing datasets")
        
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH)
        
        # Remove unnecessary columns
        cols_to_remove = ['date', 'subject', 'title', 'text']
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=cols_to_remove
        )
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with class weights."""
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'balanced_accuracy': balanced_accuracy_score(predictions, labels),
        'accuracy': accuracy_score(predictions, labels)
    }

def get_performance_metrics(df_test):
    y_test = df_test.label
    y_pred = df_test.predictions

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

def make_predictions(model, tokenizer, df_test, batch_size=128):
    logger.info("Making predictions on test set")
    
    sentences = df_test.text.tolist()
    all_outputs = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="Predicting"):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch_sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH
        )
        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
    
    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions'] = final_outputs.argmax(axis=1).cpu().numpy()
    
    return df_test

def main():
    # Load datasets
    data_loader = DatasetLoader()
    df_train, df_val, df_test = data_loader.load_datasets()
    dataset = data_loader.create_huggingface_datasets(df_train, df_val, df_test)
    class_weights = data_loader.calculate_class_weights(df_train)
    
    # Setup model and tokenizer
    trainer_instance = ModelTrainer()
    trainer_instance.setup_model()
    trainer_instance.setup_tokenizer()
    
    # Preprocess datasets
    tokenized_datasets = trainer_instance.preprocess_datasets(dataset)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir='fakenews_classification',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=1,
        report_to=None  # Disable wandb logging
    )
    
    # Initialize trainer
    collate_fn = DataCollatorWithPadding(tokenizer=trainer_instance.tokenizer)
    trainer = CustomTrainer(
        model=trainer_instance.model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=trainer_instance.tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model and metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(df_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model("saved_model")
    logger.info("Model saved successfully")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    df_test_with_predictions = make_predictions(
        trainer_instance.model, 
        trainer_instance.tokenizer, 
        df_test
    )
    
    # Save results and print metrics
    df_test_with_predictions.to_csv("df_test_with_predictions.csv", index=False)
    get_performance_metrics(df_test_with_predictions)

if __name__ == "__main__":
    main()