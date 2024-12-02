import re
import pandas as pd

def clean_recipe_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file, header=None)
    
    # Rename columns for clarity
    df.columns = [
        "Course", "Cooking Method", "Cuisine", "Image URL", 
        "Ingredients", "Prep Time (minutes)", "Recipe Name", 
        "Servings", "Tags"
    ]
    
    # Function to wrap ingredients and tags in double quotes using regex
    def apply_regex(row):
        # Wrap the ingredients field in double quotes after the .jpg link
        row[4] = re.sub(r'(?<=\.jpg,)([^,]+)(?=,)', r'"\1"', row[4])  # Ingredients
        
        # Wrap the Tags field in double quotes
        row[8] = re.sub(r'([^,]+)$', r'"\1"', row[8])  # Tags
        
        return row
    
    # Apply regex function to each row
    df = df.apply(apply_regex, axis=1)
    
    # Save cleaned data to a new file
    df.to_csv(output_file, index=False, header=False)
    print(f"Data has been cleaned and saved to {output_file}")

if __name__ == "__main__":
    # Input and output file paths
    input_csv = "cleaned_recipe_1.csv"
    output_csv = "cleaned_recipe.csv"
    
    # Run the cleaning function
    clean_recipe_data(input_csv, output_csv)
