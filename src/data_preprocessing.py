# data_preprocessing.py
import pandas as pd
import numpy as np
import logging
import csv
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load and perform initial cleaning of recipe data."""
    try:
        # Read CSV with more robust parameters for handling quotes and special characters
        df = pd.read_csv(file_path, 
                        sep=',',
                        quotechar='"',
                        escapechar='\\',
                        encoding='utf-8',
                        quoting=csv.QUOTE_ALL,  # Handle all fields as quoted
                        on_bad_lines='skip')
        
        # Define expected columns based on the dataset structure
        expected_columns = [
            'category', 'cooking_method', 'cuisine', 'image', 
            'ingredients', 'prep_time', 'recipe_name', 'serves', 'tags'
        ]
        
        # Ensure column names are clean and match expected format
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Verify all expected columns exist
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = ''
        
        # Keep only expected columns in specified order
        df = df[expected_columns]
        
        # Convert numeric columns
        df['prep_time'] = pd.to_numeric(df['prep_time'], errors='coerce').fillna(0).astype(int)
        df['serves'] = pd.to_numeric(df['serves'], errors='coerce').fillna(1).astype(int)
        
        # Fill missing values for categorical columns
        categorical_cols = ['category', 'cooking_method', 'cuisine']
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown').str.strip()
        
        logging.info(f"Successfully loaded data with shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_ingredients(df):
    """Process ingredients into clean, standardized lists."""
    try:
        df = df.copy()
        
        def clean_ingredient_list(ingredients_str):
            if pd.isna(ingredients_str):
                return []
            # Split by comma, clean each ingredient, and remove empty strings
            ingredients = [
                ing.strip().lower() 
                for ing in str(ingredients_str).split(',')
                if ing.strip()
            ]
            return ingredients
        
        df['ingredients'] = df['ingredients'].apply(clean_ingredient_list)
        logging.info("Successfully preprocessed ingredients")
        return df
    
    except Exception as e:
        logging.error(f"Error preprocessing ingredients: {str(e)}")
        raise

def encode_categorical_features(df):
    """Encode categorical variables for analysis."""
    try:
        df = df.copy()
        label_encoders = {}
        categorical_columns = ['category', 'cooking_method', 'cuisine']
        
        for column in categorical_columns:
            le = LabelEncoder()
            # Handle NaN values and convert to lowercase
            df[column] = df[column].fillna('unknown').str.lower().str.strip()
            df[f'{column}_encoded'] = le.fit_transform(df[column])
            label_encoders[column] = le
            
        logging.info("Successfully encoded categorical features")
        return df, label_encoders
    
    except Exception as e:
        logging.error(f"Error encoding categorical features: {str(e)}")
        raise

def process_tags(df):
    """Process and standardize recipe tags."""
    try:
        df = df.copy()
        
        def clean_tag_list(tags_str):
            if pd.isna(tags_str):
                return []
            # Split by comma, clean each tag, and remove empty strings
            tags = [
                tag.strip().lower() 
                for tag in str(tags_str).split(',')
                if tag.strip()
            ]
            return tags
        
        df['tags'] = df['tags'].apply(clean_tag_list)
        logging.info("Successfully processed tags")
        return df
    
    except Exception as e:
        logging.error(f"Error processing tags: {str(e)}")
        raise

def preprocess_data(file_path):
    """Complete data preprocessing pipeline."""
    try:
        # Load data
        df = load_data(file_path)
        
        # Process ingredients
        df = preprocess_ingredients(df)
        
        # Encode categorical features
        df, label_encoders = encode_categorical_features(df)
        
        # Process tags
        df = process_tags(df)
        
        logging.info("Completed full data preprocessing pipeline")
        return df, label_encoders
    
    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    data_path = "data/cleaned_recipe.csv"
    try:
        df, encoders = preprocess_data(data_path)
        
        # Print sample results
        print("\nSample processed recipe:")
        sample_recipe = df.iloc[0]
        print(f"\nRecipe Name: {sample_recipe['recipe_name']}")
        print(f"Category: {sample_recipe['category']}")
        print(f"Cuisine: {sample_recipe['cuisine']}")
        print(f"Ingredients: {sample_recipe['ingredients']}")
        print(f"Tags: {sample_recipe['tags']}")
        
        print("\nEncoded features:")
        for col in ['category', 'cooking_method', 'cuisine']:
            print(f"{col}_encoded: {sample_recipe[f'{col}_encoded']}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")