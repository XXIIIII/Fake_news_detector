# Fake News Detection using LLaMA-3

This project implements a fake news detection system using fine-tuned LLaMA-3 models with LoRA (Low-Rank Adaptation) for efficient parameter updates.

## Project Structure

```
Fake News Detection/
├── data_preprocessing.py      # Dataset preparation and splitting
├── finetune_llama3.py         # Fine-tuning script for LLaMA-3
├── baseline_model.ipynb       # Baseline model evaluation
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Features

- **Dataset Processing**: Automated data loading, labeling, and splitting
- **Model Fine-tuning**: LLaMA-3 fine-tuning with LoRA for efficiency
- **Baseline Evaluation**: Zero-shot performance testing
- **Performance Metrics**: Comprehensive evaluation with multiple metrics
- **Class Balancing**: Weighted loss function for handling class imbalance

### Baseline Model
- **Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Approach**: Zero-shot evaluation 
- **Task**: Binary Classification (Real vs Fake News)

### Fine-tuned Model
- **Base Model**: meta-llama/Meta-Llama-3-8B (non-instruct)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Binary Classification (Real vs Fake News)

## Performance

The fine-tuned model achieves 35% improvement in accuracy compared to the baseline model


