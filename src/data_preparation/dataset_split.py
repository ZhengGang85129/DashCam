import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split(ratio: float = 0.1):
    df = pd.read_csv(dataset)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
 

# Verify the sizes
    print(f"Total records: {len(df)}")
    print(f"Training set: {len(train_df)} records ({len(train_df)/len(df):.1%})")
    print(f"Validation set: {len(val_df)} records ({len(val_df)/len(df):.1%})")

    # Save to separate CSV files
    train_df.to_csv('dataset/img_database/train.csv', index=False)
    val_df.to_csv('dataset/img_database/validation.csv', index=False)
    print('Check meta data: dataset/img_database/train.csv') 
    print('Check meta data: dataset/img_database/validation.csv') 

def main():
    global dataset 
    dataset = 'dataset/train.csv'
    os.makedirs('dataset/img_database', exist_ok = True)
    split() 
    pass


if __name__ == "__main__":
    main()
