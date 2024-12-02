# recipe_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage

def load_and_preprocess_data(file_path):
    """Load and preprocess the recipe dataset."""
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert ingredients string to list
    df['ingredients'] = df['ingredients'].apply(lambda x: [i.strip().lower() for i in str(x).split(',')])
    
    return df

def create_cuisine_similarity_heatmap(df, output_path='cuisine_similarity_heatmap.png'):
    """Create a heatmap showing similarities between cuisines based on ingredients."""
    # Create cuisine-ingredient matrix
    cuisine_groups = df.groupby('cuisine')
    cuisines = sorted(df['cuisine'].unique())
    
    # Calculate similarities between cuisines
    similarity_matrix = np.zeros((len(cuisines), len(cuisines)))
    
    for i, cuisine1 in enumerate(cuisines):
        for j, cuisine2 in enumerate(cuisines):
            if i <= j:
                # Get all ingredients for each cuisine
                ingredients1 = set(df[df['cuisine'] == cuisine1]['ingredients'].explode())
                ingredients2 = set(df[df['cuisine'] == cuisine2]['ingredients'].explode())
                
                # Calculate Jaccard similarity
                intersection = len(ingredients1.intersection(ingredients2))
                union = len(ingredients1.union(ingredients2))
                similarity = intersection / union if union > 0 else 0
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
    
    # Create heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(similarity_matrix, 
                xticklabels=cuisines,
                yticklabels=cuisines,
                cmap='YlOrRd')
    plt.title('Cuisine Similarity Based on Ingredients')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_ingredient_network(df, output_path='ingredient_network.png', min_cooccurrence=10):
    """Create a network graph showing ingredient co-occurrences."""
    # Create ingredient co-occurrence matrix
    all_ingredients = []
    for ingredients in df['ingredients']:
        for pair in zip(ingredients[:-1], ingredients[1:]):
            all_ingredients.append(tuple(sorted(pair)))
    
    # Count co-occurrences
    cooccurrence = pd.Series(all_ingredients).value_counts()
    
    # Create network
    G = nx.Graph()
    
    # Add edges for ingredients that co-occur frequently
    for (ing1, ing2), count in cooccurrence.items():
        if count >= min_cooccurrence:
            G.add_edge(ing1, ing2, weight=count)
    
    # Draw network
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges with varying thickness
    edge_weights = [G[u][v]['weight'] for (u, v) in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=[w/max(edge_weights)*5 for w in edge_weights],
                          alpha=0.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Ingredient Co-occurrence Network\n(Edge thickness indicates frequency of co-occurrence)')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_cuisine_dendrogram(df, output_path='cuisine_dendrogram.png'):
    """Create a dendrogram showing cuisine clustering based on cooking methods."""
    # Create cuisine-cooking method matrix
    cuisine_method = pd.crosstab(df['cuisine'], df['cooking_method'])
    
    # Normalize
    cuisine_method = cuisine_method.div(cuisine_method.sum(axis=1), axis=0)
    
    # Calculate linkage
    Z = linkage(cuisine_method, 'ward')
    
    # Create dendrogram
    plt.figure(figsize=(15, 10))
    dendrogram(Z, labels=cuisine_method.index, leaf_rotation=90)
    plt.title('Cuisine Clustering by Cooking Methods')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_recipe_scatter(df, output_path='recipe_scatter.png'):
    """Create a scatter plot of recipes based on prep time and number of ingredients."""
    plt.figure(figsize=(12, 8))
    
    # Calculate number of ingredients for each recipe
    n_ingredients = df['ingredients'].apply(len)
    
    # Create scatter plot
    colors = df['cuisine'].astype('category').cat.codes
    plt.scatter(df['prep_time'], n_ingredients, c=colors, alpha=0.6)
    
    plt.xlabel('Preparation Time (minutes)')
    plt.ylabel('Number of Ingredients')
    plt.title('Recipe Complexity')
    
    # Add legend for cuisines
    unique_cuisines = df['cuisine'].unique()
    handles = [plt.scatter([], [], c=f'C{i}', label=cuisine) 
              for i, cuisine in enumerate(unique_cuisines)]
    plt.legend(handles=handles, title='Cuisine', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('data\cleaned_recipe.csv')
    
    # Create visualizations
    print("Creating cuisine similarity heatmap...")
    create_cuisine_similarity_heatmap(df)
    
    print("Creating ingredient network...")
    create_ingredient_network(df)
    
    print("Creating cuisine dendrogram...")
    create_cuisine_dendrogram(df)
    
    print("Creating recipe scatter plot...")
    create_recipe_scatter(df)
    
    print("All visualizations have been created!")

if __name__ == "__main__":
    main()