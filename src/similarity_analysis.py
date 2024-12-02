# similarity_analysis.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging

class RecipeSimilarityAnalyzer:
    def __init__(self, feature_matrix, recipe_names):
        """Initialize analyzer with feature matrix and recipe names."""
        self.feature_matrix = feature_matrix
        self.recipe_names = recipe_names
        self.similarity_matrix = None
        self.clusters = None
    
    def calculate_similarity_matrix(self):
        """Calculate cosine similarity matrix between recipes."""
        try:
            similarity_matrix = cosine_similarity(self.feature_matrix)
            # Ensure self-similarity is exactly 1.0
            np.fill_diagonal(similarity_matrix, 1.0)
            self.similarity_matrix = similarity_matrix
            logging.info(f"Created similarity matrix with shape {similarity_matrix.shape}")
            return similarity_matrix
        except Exception as e:
            logging.error(f"Error calculating similarity matrix: {str(e)}")
            raise

    def find_similar_recipes(self, recipe_idx, n_similar=5):
        """Find n most similar recipes for a given recipe."""
        try:
            if self.similarity_matrix is None:
                self.calculate_similarity_matrix()
            
            # Get similarities for the specified recipe
            similarities = self.similarity_matrix[recipe_idx]
            # Get indices of most similar recipes (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            similarity_scores = similarities[similar_indices]
            
            similar_recipes = {
                'recipe_name': self.recipe_names[recipe_idx],
                'similar_recipes': [
                    {
                        'name': self.recipe_names[idx],
                        'similarity': float(score),  # Convert to Python float
                        'index': int(idx)  # Convert to Python int
                    }
                    for idx, score in zip(similar_indices, similarity_scores)
                ]
            }
            return similar_recipes
        except Exception as e:
            logging.error(f"Error finding similar recipes: {str(e)}")
            raise

    def cluster_recipes(self, n_clusters=5):
        """Cluster recipes using K-means."""
        try:
            # Initialize and fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.feature_matrix)
            self.clusters = clusters
            
            # Create cluster summary
            cluster_summary = {}
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_recipes = np.array(self.recipe_names)[cluster_mask]
                cluster_summary[f'Cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'recipes': cluster_recipes.tolist()
                }
            
            logging.info(f"Created {n_clusters} recipe clusters")
            return clusters, cluster_summary
        except Exception as e:
            logging.error(f"Error clustering recipes: {str(e)}")
            raise

    def analyze_cuisine_relationships(self, df):
        """Analyze relationships between different cuisines."""
        try:
            if self.similarity_matrix is None:
                self.calculate_similarity_matrix()
            
            # Group recipes by cuisine
            cuisine_groups = df.groupby('cuisine').indices
            cuisines = sorted(list(cuisine_groups.keys()))
            
            # Calculate average similarity between cuisines
            n_cuisines = len(cuisines)
            cuisine_similarity = np.zeros((n_cuisines, n_cuisines))
            
            for i, cuisine1 in enumerate(cuisines):
                for j, cuisine2 in enumerate(cuisines):
                    if i <= j:
                        idx1 = cuisine_groups[cuisine1]
                        idx2 = cuisine_groups[cuisine2]
                        similarity = np.mean(self.similarity_matrix[np.ix_(idx1, idx2)])
                        cuisine_similarity[i, j] = similarity
                        cuisine_similarity[j, i] = similarity
            
            # Create DataFrame with cuisine similarities
            cuisine_similarity_df = pd.DataFrame(
                cuisine_similarity,
                index=cuisines,
                columns=cuisines
            )
            
            logging.info("Successfully analyzed cuisine relationships")
            return cuisine_similarity_df
        except Exception as e:
            logging.error(f"Error analyzing cuisine relationships: {str(e)}")
            raise

def analyze_recipe_similarities(df, feature_matrix):
    """Main function to perform recipe similarity analysis."""
    try:
        # Initialize analyzer
        analyzer = RecipeSimilarityAnalyzer(feature_matrix, df['recipe_name'].tolist())
        
        # Calculate similarity matrix
        similarity_matrix = analyzer.calculate_similarity_matrix()
        
        # Perform clustering
        clusters, cluster_summary = analyzer.cluster_recipes()
        
        # Analyze cuisine relationships
        cuisine_similarities = analyzer.analyze_cuisine_relationships(df)
        
        # Create summary statistics
        stats = {
            'n_recipes': len(df),
            'n_cuisines': df['cuisine'].nunique(),
            'n_categories': df['category'].nunique(),
            'avg_similarity': float(np.mean(similarity_matrix)),
            'cuisines': df['cuisine'].unique().tolist()
        }
        
        # Combine results
        results = {
            'similarity_matrix': similarity_matrix,
            'clusters': clusters,
            'cluster_summary': cluster_summary,
            'cuisine_similarities': cuisine_similarities,
            'stats': stats
        }
        
        logging.info("Successfully completed similarity analysis")
        return results
    except Exception as e:
        logging.error(f"Error in recipe similarity analysis: {str(e)}")
        raise