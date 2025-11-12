"""
Baseline Model Evaluation for Fake News Detection

This script evaluates the zero-shot performance of LLaMA-3-8B-Instruct
on fake news detection without any fine-tuning. It serves as a baseline
to compare against the fine-tuned model performance.
"""

import re
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report, 
    balanced_accuracy_score, accuracy_score
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineNewsClassifier:
    """
    Baseline fake news classifier using LLaMA-3-8B-Instruct
    without any fine-tuning.
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.messages = None
        self.generation_args = None
        self._setup_model()
        self._setup_chat_template()
        
    def _setup_model(self):
        """Load and configure the pre-trained model."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type='nf4'
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configure generation parameters
        self.generation_args = {
            "max_new_tokens": 256,
            "temperature": 0.01,
            "do_sample": True,
        }
        
        logger.info("Model setup complete")
    
    def _setup_chat_template(self):
        self.messages = [
            {
                "role": "system", 
                "content": "Your task is to distinguish fake news. Output True or False only."
            }
        ]
    
    @torch.no_grad()
    def classify_text(self, text: str) -> Optional[int]:
        """
        Classify a single news article as real or fake.
        """
        try:
            # Prepare chat messages
            chat_messages = self.messages + [{"role": "user", "content": text}]
            
            # Apply chat template
            token_sentence = self.tokenizer.apply_chat_template(
                chat_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize input
            input_tokens = self.tokenizer(token_sentence, return_tensors='pt').to('cuda')
            
            # Generate response
            output_tokens = self.model.generate(**input_tokens, **self.generation_args)
            
            # Decode output
            decoded_output = self.tokenizer.batch_decode(output_tokens)
            response = decoded_output[0].split('<|end_header_id|>')[-1]
            
            # Extract prediction from response
            return self._extract_prediction(response)
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return 2  # Return uncertain for errors
    
    def _extract_prediction(self, response: str) -> int:
        """
        Extract binary prediction from model response.
        """
        match = re.search(r'\b(True|False)\b', response)
        if match:
            if match.group(0) == 'True':
                return 0  # Real news
            elif match.group(0) == 'False':
                return 1  # Fake news
        return 2  # Uncertain/No clear answer
    
    def classify_batch(self, texts: List[str]) -> List[int]:
        """
        Classify a batch of news articles.
        """
        predictions = []
        
        for text in tqdm(texts, desc="Classifying articles"):
            prediction = self.classify_text(text)
            predictions.append(prediction)
            
        return predictions

class DatasetLoader:
    @staticmethod
    def load_test_dataset(test_ratio: float = 0.2, random_state: int = 1) -> pd.DataFrame:
        """
        Load and prepare test dataset from True.csv and Fake.csv files.
        """
        try:
            # Load datasets
            df_true = pd.read_csv("True.csv")
            df_true['label'] = 0  # True news
            
            df_false = pd.read_csv("Fake.csv")
            df_false['label'] = 1  # Fake news
            
            # Combine and shuffle
            df = pd.concat([df_true, df_false], ignore_index=True)
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Extract test portion
            val_end_point = int(df.shape[0] * (1 - test_ratio))
            df_test = df.iloc[val_end_point:, :].reset_index(drop=True)
            
            logger.info(f"Test dataset size: {len(df_test)} articles")
            return df_test
            
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise

class EvaluationMetrics:
    @staticmethod
    def compute_performance_metrics(df_test: pd.DataFrame, prediction_col: str = 'predictions'):
        """
        Compute and display comprehensive performance metrics.
        """
        y_true = df_test['label']
        y_pred = df_test[prediction_col]
        
        # Filter out uncertain predictions (label 2) for metrics calculation
        mask = y_pred != 2
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        print("=== BASELINE MODEL PERFORMANCE ===")
        print(f"Total samples: {len(df_test)}")
        print(f"Uncertain predictions: {sum(y_pred == 2)} ({sum(y_pred == 2)/len(df_test)*100:.1f}%)")
        print(f"Evaluated samples: {len(y_true_filtered)}")
        print()
        
        if len(y_true_filtered) > 0:
            print("Confusion Matrix:")
            print(confusion_matrix(y_true_filtered, y_pred_filtered))
            print()
            
            print("Classification Report:")
            print(classification_report(y_true_filtered, y_pred_filtered, 
                                      target_names=['Real News', 'Fake News']))
            print()
            
            print("Performance Metrics:")
            print(f"Accuracy Score: {accuracy_score(y_true_filtered, y_pred_filtered):.4f}")
            print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_true_filtered, y_pred_filtered):.4f}")
            print(f"F1 Score: {f1_score(y_true_filtered, y_pred_filtered, average='weighted'):.4f}")
        else:
            print("No valid predictions to evaluate!")

def save_results(df_test: pd.DataFrame, filename: str = "baseline_results.csv"):
    df_test.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")

def main():
    df_test = DatasetLoader.load_test_dataset()
    
    # Initialize classifier
    classifier = BaselineNewsClassifier()
    
    # Make predictions
    logger.info("Making predictions on test set...")
    predictions = classifier.classify_batch(df_test['text'].tolist())
    df_test['predictions'] = predictions
    
    # Evaluate performance
    EvaluationMetrics.compute_performance_metrics(df_test)
    
    # Save results
    save_results(df_test)
    
    logger.info("Baseline evaluation complete!")

if __name__ == "__main__":
    main()