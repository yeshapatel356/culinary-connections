import pandas as pd
import os

# Load dataset (adjust file path as per your dataset location)
dataset_path = "C:\\Users\\patel\\.cache\\kagglehub\\datasets\\snehallokesh31096\\recipe\\versions\\1\\recipes_82k.csv"
df = pd.read_csv(dataset_path)

# Check the first few rows of the dataset to inspect columns and data
print("First few rows of the dataset:")
print(df.head())

# Check the data types of the columns
print("\nData types of columns:")
print(df.dtypes)

# Preprocess the data:
# For ingredients, let's clean the text (convert to lowercase and remove non-alphabetical characters)
df['ingredients'] = df['ingredients'].str.lower().str.replace('[^a-zA-Z\s]', '', regex=True)

# Check if 'cooking_method' column exists and process it as steps
if 'cooking_method' in df.columns:
    # Clean 'cooking_method' column (remove non-alphabetic characters, lowercased)
    df['cooking_method'] = df['cooking_method'].str.lower().str.replace('[^a-zA-Z\s]', '', regex=True)
else:
    print("No 'cooking_method' column found in the dataset. Skipping cooking_method preprocessing.")

# Drop rows with missing values in important columns (i.e., ingredients and cooking_method)
df = df.dropna(subset=['ingredients', 'cooking_method'])

# Save the cleaned dataset into the "data" folder
cleaned_data_path = "data/cleaned_recipes.csv"
df.to_csv(cleaned_data_path, index=False)
print(f"\nCleaned data saved to {cleaned_data_path}")
