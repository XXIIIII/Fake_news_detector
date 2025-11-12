import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_datasets():
    try:
        # Load datasets
        df_true = pd.read_csv("True.csv")
        df_false = pd.read_csv("Fake.csv")
        
        # Assign labels
        df_true['label'] = 0  # True news
        df_false['label'] = 1  # Fake news
        
        logger.info(f"Loaded {len(df_true)} true news articles")
        logger.info(f"Loaded {len(df_false)} fake news articles")
        
        return df_true, df_false
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        raise

def combine_and_shuffle_data(df_true, df_false, random_state=1):
    """
    Combine true and fake news datasets and shuffle the data.
    """
    df_combined = pd.concat([df_true, df_false], ignore_index=True)
    df_shuffled = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Combined dataset size: {len(df_shuffled)} articles")
    return df_shuffled

def split_dataset(df, train_ratio=0.6, val_ratio=0.2):
    """
    Split dataset into training, validation, and test sets.
    """
    total_size = df.shape[0]
    
    # Calculate split points
    train_end_point = int(total_size * train_ratio)
    val_end_point = int(total_size * (train_ratio + val_ratio))
    
    # Split data
    df_train = df.iloc[:train_end_point, :]
    df_val = df.iloc[train_end_point:val_end_point, :]
    df_test = df.iloc[val_end_point:, :]
    
    logger.info(f"Train set size: {len(df_train)} ({train_ratio*100:.1f}%)")
    logger.info(f"Validation set size: {len(df_val)} ({val_ratio*100:.1f}%)")
    logger.info(f"Test set size: {len(df_test)} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    return df_train, df_val, df_test

def save_datasets(df_train, df_val, df_test):
    """
    Save the split datasets to CSV files.
    """
    df_train.to_csv("df_train.csv", index=False)
    df_val.to_csv("df_val.csv", index=False)
    df_test.to_csv("df_test.csv", index=False)
    logger.info("Datasets saved successfully")

def main():
    df_true, df_false = load_and_prepare_datasets()
    df_combined = combine_and_shuffle_data(df_true, df_false)
    df_train, df_val, df_test = split_dataset(df_combined)
    
    # Save datasets
    save_datasets(df_train, df_val, df_test)

if __name__ == "__main__":
    main()