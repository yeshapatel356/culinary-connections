import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load and perform initial cleaning of recipe data."""
    try:
        # Use more robust CSV reading parameters
        df = pd.read_csv(file_path, 
                        escapechar='\\',
                        quoting=1,  # QUOTE_ALL
                        encoding='utf-8',
                        on_bad_lines='skip')
        logging.info(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_ingredients(df):
    """Convert ingredients string to list and clean ingredients."""
    try:
        # Handle potential NaN values
        df['ingredients'] = df['ingredients'].fillna('')
        
        # Split ingredients and clean
        df['ingredients'] = df['ingredients'].apply(lambda x: [
            i.strip().lower() for i in str(x).split(',') if i.strip()
        ])
        
        logging.info("Successfully preprocessed ingredients")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing ingredients: {str(e)}")
        raise

def encode_categorical_features(df):
    """Encode categorical variables for analysis."""
    try:
        label_encoders = {}
        categorical_columns = ['category', 'cooking_method', 'cuisine']
        
        for column in categorical_columns:
            if column in df.columns:
                le = LabelEncoder()
                # Handle NaN values
                df[column] = df[column].fillna('unknown')
                df[f'{column}_encoded'] = le.fit_transform(df[column])
                label_encoders[column] = le
        
        logging.info("Successfully encoded categorical features")
        return df, label_encoders
    except Exception as e:
        logging.error(f"Error encoding categorical features: {str(e)}")
        raise

def process_tags(df):
    """Process and encode recipe tags."""
    try:
        # Handle NaN values in tags
        df['tags'] = df['tags'].fillna('')
        df['tags'] = df['tags'].apply(lambda x: [
            tag.strip().lower() for tag in str(x).split(',') if tag.strip()
        ])
        logging.info("Successfully processed tags")
        return df
    except Exception as e:
        logging.error(f"Error processing tags: {str(e)}")
        raise