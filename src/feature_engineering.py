# # feature_engineering.py
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import logging

# def create_ingredient_features(df):
#     """Create TF-IDF features from ingredients."""
#     try:
#         # Convert ingredients lists to strings for TF-IDF
#         ingredients_str = df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
#         # Create TF-IDF features with reduced dimensions
#         tfidf = TfidfVectorizer(max_features=50, stop_words='english')
#         ingredient_features = tfidf.fit_transform(ingredients_str)
        
#         logging.info(f"Created ingredient features with shape {ingredient_features.shape}")
#         return ingredient_features, tfidf
#     except Exception as e:
#         logging.error(f"Error creating ingredient features: {str(e)}")
#         raise

# def create_cooking_features(df):
#     """Create numerical features for cooking-related attributes."""
#     try:
#         scaler = MinMaxScaler()
#         # Handle potential NaN values
#         cooking_features = df[['prep_time', 'serves']].fillna(0)
#         cooking_features = scaler.fit_transform(cooking_features)
        
#         logging.info(f"Created cooking features with shape {cooking_features.shape}")
#         return cooking_features, scaler
#     except Exception as e:
#         logging.error(f"Error creating cooking features: {str(e)}")
#         raise

# def create_cuisine_features(df):
#     """Create one-hot encoded features for cuisine types."""
#     try:
#         df['cuisine'] = df['cuisine'].fillna('unknown')
#         cuisine_features = pd.get_dummies(df['cuisine'], prefix='cuisine')
        
#         logging.info(f"Created cuisine features with shape {cuisine_features.shape}")
#         return cuisine_features
#     except Exception as e:
#         logging.error(f"Error creating cuisine features: {str(e)}")
#         raise

# def combine_features(ingredient_features, cooking_features, cuisine_features):
#     """Combine all features into a single matrix."""
#     try:
#         # Convert sparse matrix to dense for ingredient features
#         ingredient_dense = ingredient_features.toarray()
        
#         # Combine all features
#         combined_features = np.hstack([
#             ingredient_dense,
#             cooking_features,
#             cuisine_features.values
#         ])
        
#         logging.info(f"Combined features shape: {combined_features.shape}")
#         return combined_features
#     except Exception as e:
#         logging.error(f"Error combining features: {str(e)}")
#         raise

# feature_engineering.py
# feature_engineering.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import logging

def create_ingredient_features(df):
    """Create TF-IDF features from ingredients."""
    try:
        # Convert ingredients lists to strings
        df['ingredients_str'] = df['ingredients'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x)
        )
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        ingredient_features = tfidf.fit_transform(df['ingredients_str'])
        
        logging.info(f"Created ingredient features with shape {ingredient_features.shape}")
        return ingredient_features.toarray()
    except Exception as e:
        logging.error(f"Error creating ingredient features: {str(e)}")
        raise

def create_cooking_features(df):
    """Create numerical features for cooking-related attributes."""
    try:
        features = pd.DataFrame(index=df.index)
        
        # Prep time feature
        features['prep_time'] = pd.to_numeric(df['prep_time'], errors='coerce').fillna(0)
        
        # Serves feature
        features['serves'] = pd.to_numeric(df['serves'], errors='coerce').fillna(0)
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        logging.info(f"Created cooking features with shape {scaled_features.shape}")
        return scaled_features
    except Exception as e:
        logging.error(f"Error creating cooking features: {str(e)}")
        raise

def create_categorical_features(df):
    """Create one-hot encoded features for categorical variables."""
    try:
        # Fill missing values
        df_copy = df.copy()
        categorical_cols = ['cuisine', 'cooking_method', 'category']
        for col in categorical_cols:
            df_copy[col] = df_copy[col].fillna('unknown')
        
        # Create one-hot encoding
        categorical_features = pd.get_dummies(
            df_copy[categorical_cols],
            prefix=categorical_cols
        )
        
        logging.info(f"Created categorical features with shape {categorical_features.shape}")
        return categorical_features.values
    except Exception as e:
        logging.error(f"Error creating categorical features: {str(e)}")
        raise

def create_tag_features(df):
    """Create features from recipe tags."""
    try:
        # Get all unique tags
        all_tags = set()
        for tags_list in df['tags']:
            if isinstance(tags_list, list):
                all_tags.update(tag.lower().strip() for tag in tags_list)
        
        # Create tag matrix
        tag_list = sorted(list(all_tags))
        tag_matrix = np.zeros((len(df), len(tag_list)))
        
        # Fill matrix
        for i, tags in enumerate(df['tags']):
            if isinstance(tags, list):
                for tag in tags:
                    if tag.lower().strip() in all_tags:
                        tag_idx = tag_list.index(tag.lower().strip())
                        tag_matrix[i, tag_idx] = 1
        
        logging.info(f"Created tag features with shape {tag_matrix.shape}")
        return tag_matrix
        
    except Exception as e:
        logging.error(f"Error creating tag features: {str(e)}")
        raise

def create_feature_matrix(df):
    """Combine all features into a single matrix."""
    try:
        logging.info(f"Processing {len(df)} recipes")
        
        # Create individual feature sets
        ingredient_features = create_ingredient_features(df)
        logging.info(f"Ingredient features shape: {ingredient_features.shape}")
        
        cooking_features = create_cooking_features(df)
        logging.info(f"Cooking features shape: {cooking_features.shape}")
        
        categorical_features = create_categorical_features(df)
        logging.info(f"Categorical features shape: {categorical_features.shape}")
        
        tag_features = create_tag_features(df)
        logging.info(f"Tag features shape: {tag_features.shape}")
        
        # Verify dimensions
        expected_rows = len(df)
        for features, name in [
            (ingredient_features, 'ingredient'),
            (cooking_features, 'cooking'),
            (categorical_features, 'categorical'),
            (tag_features, 'tag')
        ]:
            actual_rows = features.shape[0]
            if actual_rows != expected_rows:
                raise ValueError(f"Dimension mismatch: {name} features have {actual_rows} rows, expected {expected_rows}")
        
        # Combine features
        combined_features = np.hstack([
            ingredient_features,
            cooking_features,
            categorical_features,
            tag_features
        ])
        
        logging.info(f"Created combined feature matrix with shape {combined_features.shape}")
        return combined_features
        
    except Exception as e:
        logging.error(f"Error creating feature matrix: {str(e)}")
        raise

def normalize_features(feature_matrix):
    """Normalize the combined feature matrix."""
    try:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        logging.info("Successfully normalized feature matrix")
        return normalized_features
    except Exception as e:
        logging.error(f"Error normalizing features: {str(e)}")
        raise
