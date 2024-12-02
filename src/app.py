# app.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from data_preprocessing import load_data, preprocess_ingredients, encode_categorical_features, process_tags
from feature_engineering import create_feature_matrix, normalize_features
from visualization import generate_visualizations
from similarity_analysis import analyze_recipe_similarities

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'results', 'visualizations']
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory ready: {dir_path.absolute()}")

def main():
    """Main execution function."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Setup directories
        setup_directories()
        
        # Load and preprocess data
        data_path = "data/cleaned_recipe.csv"
        logging.info(f"Loading data from {data_path}")
        
        # Load raw data
        df = load_data(data_path)
        logging.info(f"Loaded {len(df)} recipes")
        
        # Preprocess data
        df = preprocess_ingredients(df)
        df, label_encoders = encode_categorical_features(df)
        df = process_tags(df)
        
        # Create feature matrix
        logging.info("Creating feature matrix...")
        feature_matrix = create_feature_matrix(df)
        
        # Normalize features
        logging.info("Normalizing features...")
        normalized_features = normalize_features(feature_matrix)
        
        # Analyze similarities
        logging.info("Analyzing recipe similarities...")
        results = analyze_recipe_similarities(df, normalized_features)
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        vis_dir = generate_visualizations(df, results['similarity_matrix'], results['clusters'])
        logging.info(f"Visualizations saved to {vis_dir}")
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save matrices
        np.save(results_dir / 'similarity_matrix.npy', results['similarity_matrix'])
        np.save(results_dir / 'feature_matrix.npy', normalized_features)
        
        # Save processed data
        df.to_csv(results_dir / 'processed_recipes.csv', index=False)
        
        # Save summaries
        summary_file = results_dir / 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Recipe Analysis Summary\n")
            f.write("=====================\n\n")
            f.write(f"Total recipes analyzed: {len(df)}\n")
            f.write(f"Number of cuisines: {df['cuisine'].nunique()}\n")
            f.write(f"Number of categories: {df['category'].nunique()}\n")
            f.write(f"Number of cooking methods: {df['cooking_method'].nunique()}\n")
            f.write("\nCluster Summary:\n")
            for cluster_id, details in results['cluster_summary'].items():
                f.write(f"\n{cluster_id}:\n")
                f.write(f"  Size: {details['size']}\n")
                f.write(f"  Sample recipes: {', '.join(details['recipes'][:5])}\n")
        
        logging.info("Analysis completed successfully")
        return df, results
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        df, results = main()
        
        # Print summary
        print("\nAnalysis Complete!")
        print(f"Processed {len(df)} recipes")
        print(f"Created {len(results['cluster_summary'])} recipe clusters")
        print("\nResults saved in:")
        print("- results/processed_recipes.csv")
        print("- results/analysis_summary.txt")
        print("- results/similarity_matrix.npy")
        print("- results/feature_matrix.npy")
        print("\nVisualizations available in:")
        print("- visualizations/index.html")
        
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise