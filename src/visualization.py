
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import networkx as nx
# from scipy.cluster.hierarchy import dendrogram, linkage
# import logging

# def create_focused_cuisine_network(df, similarity_matrix, vis_dir, min_recipes=3):
#     """Create an interconnected network of cuisine relationships."""
#     try:
#         # Filter for cuisines with significant number of recipes
#         cuisine_counts = df['cuisine'].value_counts()
#         major_cuisines = cuisine_counts[cuisine_counts >= min_recipes].index
        
#         # Create filtered dataframe
#         df_filtered = df[df['cuisine'].isin(major_cuisines)]
        
#         # Calculate cuisine similarities based on ingredients and cooking methods
#         cuisine_groups = df_filtered.groupby('cuisine')
#         cuisines = list(major_cuisines)
#         n_cuisines = len(cuisines)
        
#         # Initialize similarity matrix
#         cuisine_similarity = np.zeros((n_cuisines, n_cuisines))
        
#         def extract_ingredients(ingredients_str):
#             """Extract ingredients from a string."""
#             try:
#                 if not ingredients_str or pd.isna(ingredients_str).any():
#                     return set()
#                 if isinstance(ingredients_str, str):
#                     return {ing.strip().lower() for ing in ingredients_str.split(',') if ing.strip()}
#                 elif isinstance(ingredients_str, (list, np.ndarray)):
#                     return {str(ing).strip().lower() for ing in ingredients_str if str(ing).strip()}
#                 return set()
#             except Exception as e:
#                 print(f"Error processing ingredients: {e}")
#                 return set()

#         def get_cuisine_ingredients(cuisine_data):
#             """Get all ingredients for a cuisine."""
#             all_ingredients = set()
#             for ingredients in cuisine_data['ingredients'].values:
#                 all_ingredients.update(extract_ingredients(ingredients))
#             return all_ingredients

#         # Calculate similarities based on shared ingredients and cooking methods
#         for i, cuisine1 in enumerate(cuisines):
#             cuisine1_data = cuisine_groups.get_group(cuisine1)
#             ingredients1 = get_cuisine_ingredients(cuisine1_data)
            
#             for j, cuisine2 in enumerate(cuisines):
#                 if i < j:
#                     cuisine2_data = cuisine_groups.get_group(cuisine2)
#                     ingredients2 = get_cuisine_ingredients(cuisine2_data)
                    
#                     # Calculate Jaccard similarity for ingredients
#                     union_size = len(ingredients1.union(ingredients2))
#                     if union_size > 0:
#                         ingredient_similarity = len(ingredients1.intersection(ingredients2)) / union_size
#                     else:
#                         ingredient_similarity = 0
                    
#                     # Calculate similarity based on cooking methods
#                     methods1 = set(cuisine1_data['cooking_method'].dropna())
#                     methods2 = set(cuisine2_data['cooking_method'].dropna())
#                     method_union_size = len(methods1.union(methods2))
                    
#                     if method_union_size > 0:
#                         method_similarity = len(methods1.intersection(methods2)) / method_union_size
#                     else:
#                         method_similarity = 0
                    
#                     # Combine similarities with weights
#                     combined_similarity = (0.7 * ingredient_similarity + 0.3 * method_similarity)
                    
#                     cuisine_similarity[i, j] = combined_similarity
#                     cuisine_similarity[j, i] = combined_similarity
        
#         # Create network
#         G = nx.Graph()
        
#         # Add nodes with recipe counts
#         for cuisine in cuisines:
#             recipe_count = len(cuisine_groups.get_group(cuisine))
#             G.add_node(cuisine, size=recipe_count)
        
#         # Add edges for strong connections
#         # Find non-zero similarities and calculate threshold
#         nonzero_similarities = cuisine_similarity[cuisine_similarity > 0]
#         if len(nonzero_similarities) > 0:
#             threshold = np.percentile(nonzero_similarities, 25)  # Use lower percentile
#         else:
#             threshold = 0.1  # Default threshold if no similarities found
        
#         # Add edges
#         for i, cuisine1 in enumerate(cuisines):
#             for j, cuisine2 in enumerate(cuisines):
#                 if i < j and cuisine_similarity[i, j] > threshold:
#                     G.add_edge(cuisine1, cuisine2, weight=cuisine_similarity[i, j])
        
#         # Create visualization with improved layout
#         plt.figure(figsize=(15, 12))
        
#         # Use spring_layout if kamada_kawai fails
#         try:
#             pos = nx.kamada_kawai_layout(G)
#         except:
#             pos = nx.spring_layout(G, k=2, iterations=50)
        
#         # Draw edges with varying thickness and transparency
#         if G.number_of_edges() > 0:  # Only draw edges if they exist
#             edge_weights = [G[u][v]['weight'] * 5 for (u, v) in G.edges()]
#             nx.draw_networkx_edges(G, pos, 
#                                  width=edge_weights,
#                                  alpha=0.4,
#                                  edge_color='gray',
#                                  style='solid')
        
#         # Draw nodes with size based on recipe count and better colors
#         node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]
#         node_colors = plt.cm.Set3(np.linspace(0, 1, len(G.nodes())))
        
#         nx.draw_networkx_nodes(G, pos,
#                              node_size=node_sizes,
#                              node_color=node_colors,
#                              alpha=0.7)
        
#         # Add labels with better formatting
#         nx.draw_networkx_labels(G, pos,
#                               font_size=12,
#                               font_weight='bold',
#                               font_family='sans-serif')
        
#         plt.title("Cuisine Relationship Network\nNode size represents number of recipes\nEdges show ingredient and cooking method similarities",
#                  fontsize=14,
#                  pad=20)
        
#         plt.axis('off')
#         plt.tight_layout()
        
#         # Save with high quality
#         plt.savefig(vis_dir / 'cuisine_network.png',
#                    dpi=300,
#                    bbox_inches='tight',
#                    facecolor='white',
#                    edgecolor='none')
#         plt.close()
        
#     except Exception as e:
#         print(f"Error creating cuisine network: {str(e)}")
#         logging.error(f"Error in create_focused_cuisine_network: {str(e)}")
#         raise
# def create_top_ingredients_heatmap(df, vis_dir, top_n=15):
#     """Create a focused heatmap of most common ingredients across major cuisines."""
#     try:
#         # Process ingredients
#         all_ingredients = []
#         for ingredients in df['ingredients']:
#             if isinstance(ingredients, str):
#                 ingredients = [ing.strip() for ing in ingredients.split(',')]
#             if isinstance(ingredients, list):
#                 all_ingredients.extend([ing.strip().lower() for ing in ingredients])
        
#         # Get top ingredients
#         ingredient_counts = pd.Series(all_ingredients).value_counts()
#         top_ingredients = ingredient_counts.head(top_n).index
        
#         # Filter for major cuisines (those with at least 3 recipes)
#         cuisine_counts = df['cuisine'].value_counts()
#         major_cuisines = cuisine_counts[cuisine_counts >= 3].index
        
#         # Create cuisine-ingredient matrix
#         cuisine_ingredient_matrix = pd.DataFrame(0, 
#                                               index=major_cuisines,
#                                               columns=top_ingredients)
        
#         # Fill the matrix
#         for cuisine in major_cuisines:
#             cuisine_recipes = df[df['cuisine'] == cuisine]
#             for ingredient in top_ingredients:
#                 count = 0
#                 for recipe_ingredients in cuisine_recipes['ingredients']:
#                     if isinstance(recipe_ingredients, str):
#                         recipe_ingredients = [ing.strip().lower() for ing in recipe_ingredients.split(',')]
#                     if isinstance(recipe_ingredients, list):
#                         if any(ingredient in ing.lower() for ing in recipe_ingredients):
#                             count += 1
#                 cuisine_ingredient_matrix.loc[cuisine, ingredient] = count
        
#         # Normalize by cuisine
#         cuisine_ingredient_matrix = cuisine_ingredient_matrix.div(cuisine_ingredient_matrix.sum(axis=1), axis=0)
        
#         # Create heatmap
#         plt.figure(figsize=(12, 8))
#         sns.heatmap(cuisine_ingredient_matrix,
#                    cmap='YlOrRd',
#                    xticklabels=True,
#                    yticklabels=True,
#                    cbar_kws={'label': 'Relative Frequency'})
        
#         plt.title('Top Ingredients Across Major Cuisines', pad=20)
#         plt.xlabel('Ingredients', labelpad=10)
#         plt.ylabel('Cuisines', labelpad=10)
        
#         # Rotate x-axis labels
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
        
#         plt.tight_layout()
#         plt.savefig(vis_dir / 'ingredient_heatmap.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#     except Exception as e:
#         print(f"Error creating ingredient heatmap: {str(e)}")

# def create_simple_cooking_dendrogram(df, vis_dir):
#     """Create a simplified dendrogram of cuisine clustering."""
#     try:
#         # Create cuisine-cooking method matrix
#         cuisine_method = pd.crosstab(df['cuisine'], df['cooking_method'])
        
#         # Filter for cuisines with at least 3 recipes
#         cuisine_method = cuisine_method[cuisine_method.sum(axis=1) >= 3]
        
#         # Normalize
#         cuisine_method = cuisine_method.div(cuisine_method.sum(axis=1), axis=0)
        
#         # Calculate linkage
#         Z = linkage(cuisine_method, 'ward')
        
#         # Create dendrogram
#         plt.figure(figsize=(12, 8))
#         dendrogram(Z, 
#                   labels=cuisine_method.index,
#                   leaf_rotation=45,
#                   leaf_font_size=10)
        
#         plt.title('Cuisine Clustering by Cooking Methods', pad=20)
#         plt.ylabel('Distance')
        
#         plt.tight_layout()
#         plt.savefig(vis_dir / 'cooking_dendrogram.png', dpi=300, bbox_inches='tight')
#         plt.close()
        
#     except Exception as e:
#         print(f"Error creating cooking dendrogram: {str(e)}")

# def generate_visualizations(df, similarity_matrix, clusters):
#     """Generate all visualizations with improved clarity."""
#     try:
#         # Create visualizations directory
#         vis_dir = Path('visualizations')
#         vis_dir.mkdir(exist_ok=True)
        
#         # Create focused visualizations
#         create_focused_cuisine_network(df, similarity_matrix, vis_dir)
#         create_top_ingredients_heatmap(df, vis_dir)
#         create_simple_cooking_dendrogram(df, vis_dir)
        
#         # Create simple HTML index
#         html_content = """
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Recipe Analysis</title>
#             <style>
#                 body { font-family: Arial; margin: 20px; max-width: 1200px; margin: auto; }
#                 .viz { margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px; }
#                 img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
#                 h2 { color: #333; }
#             </style>
#         </head>
#         <body>
#             <h1>Recipe Analysis Visualizations</h1>
            
#             <div class="viz">
#                 <h2>Cuisine Relationships</h2>
#                 <img src="cuisine_network.png" alt="Cuisine Network">
#             </div>
            
#             <div class="viz">
#                 <h2>Common Ingredients</h2>
#                 <img src="ingredient_heatmap.png" alt="Ingredient Heatmap">
#             </div>
            
#             <div class="viz">
#                 <h2>Cuisine Clustering</h2>
#                 <img src="cooking_dendrogram.png" alt="Cooking Method Dendrogram">
#             </div>
#         </body>
#         </html>
#         """
        
#         with open(vis_dir / 'index.html', 'w') as f:
#             f.write(html_content)
        
#         return vis_dir
        
#     except Exception as e:
#         print(f"Error generating visualizations: {str(e)}")
#         raise

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster  

fig = go.Figure()
import logging
def create_interactive_cuisine_network(df, similarity_matrix, vis_dir):
    """Create an interactive network visualization of cuisine relationships."""
    try:
        logging.info("Starting cuisine network visualization...")
        
        # Filter and prepare data
        cuisine_counts = df['cuisine'].value_counts()
        major_cuisines = cuisine_counts[cuisine_counts >= 3].index
        df_filtered = df[df['cuisine'].isin(major_cuisines)]
        cuisine_groups = df_filtered.groupby('cuisine')
        cuisines = list(major_cuisines)
        n_cuisines = len(cuisines)
        
        # Calculate similarities
        cuisine_similarity = np.zeros((n_cuisines, n_cuisines))
        for i, cuisine1 in enumerate(cuisines):
            for j, cuisine2 in enumerate(cuisines):
                if i < j:
                    # Use provided similarity matrix
                    cuisine_similarity[i, j] = similarity_matrix[i, j]
                    cuisine_similarity[j, i] = similarity_matrix[i, j]
        
        # Create network
        G = nx.Graph()
        
        # Add nodes with recipe counts
        for cuisine in cuisines:
            recipe_count = len(cuisine_groups.get_group(cuisine))
            G.add_node(cuisine, size=recipe_count)
        
        # Add edges for strong connections
        threshold = np.percentile(cuisine_similarity[cuisine_similarity > 0], 25)
        for i, cuisine1 in enumerate(cuisines):
            for j, cuisine2 in enumerate(cuisines):
                if i < j and cuisine_similarity[i, j] > threshold:
                    G.add_edge(cuisine1, cuisine2, weight=cuisine_similarity[i, j])
        
        # Create layout with more spacing
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces with better styling
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(
                        width=weight * 25,  # Increased line width
                        color='rgba(169, 169, 169, 0.6)'  # More visible edges
                    ),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        # Create node trace with better styling
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        recipe_counts = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            recipe_count = G.nodes[node]['size']
            recipe_counts.append(recipe_count)
            node_text.append(f"<b>{node}</b><br>Recipes: {recipe_count}")
            node_sizes.append(recipe_count * 5)  # Adjusted size scaling
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=recipe_counts,  # Color nodes based on recipe count
                colorscale='Viridis',
                line=dict(color='white', width=2),
                showscale=True,
                colorbar=dict(title="Recipe Count")
            )
        )
        
        # Create figure with improved layout
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=dict(
                text='Cuisine Relationship Network<br><sup>Node size and color represent number of recipes, edges show similarities</sup>',
                x=0.5,
                y=0.95
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1000,
            height=800
        )
        
        # Save interactive version
        fig.write_html(vis_dir / 'cuisine_network.html')
        logging.info("Cuisine network visualization completed")
        
    except Exception as e:
        logging.error(f"Error in cuisine network visualization: {str(e)}")
        raise

def create_interactive_ingredients_heatmap(df, vis_dir, top_n=15):
    """Create an interactive heatmap of ingredients across cuisines."""
    try:
        logging.info("Starting ingredients heatmap visualization...")
        
        # Process ingredients and get unique ingredients
        all_ingredients = []
        for ingredients in df['ingredients']:
            if isinstance(ingredients, str):
                ingredients_list = [ing.strip().lower() for ing in ingredients.split(',')]
                all_ingredients.extend(ingredients_list)
            elif isinstance(ingredients, list):
                ingredients_list = [ing.strip().lower() for ing in ingredients]
                all_ingredients.extend(ingredients_list)
        
        # Get top ingredients
        ingredient_counts = pd.Series(all_ingredients).value_counts()
        top_ingredients = ingredient_counts.head(top_n).index
        
        # Filter for major cuisines
        cuisine_counts = df['cuisine'].value_counts()
        major_cuisines = cuisine_counts[cuisine_counts >= 3].index
        
        # Create cuisine-ingredient matrix
        cuisine_ingredient_matrix = pd.DataFrame(0, 
                                              index=major_cuisines,
                                              columns=top_ingredients)
        
        # Fill the matrix
        for cuisine in major_cuisines:
            cuisine_recipes = df[df['cuisine'] == cuisine]
            for ingredient in top_ingredients:
                count = 0
                for _, row in cuisine_recipes.iterrows():
                    if isinstance(row['ingredients'], str):
                        recipe_ingredients = [ing.strip().lower() for ing in row['ingredients'].split(',')]
                    elif isinstance(row['ingredients'], list):
                        recipe_ingredients = [ing.strip().lower() for ing in row['ingredients']]
                    else:
                        continue
                        
                    if any(ingredient in ing for ing in recipe_ingredients):
                        count += 1
                cuisine_ingredient_matrix.loc[cuisine, ingredient] = count
        
        # Normalize by cuisine
        cuisine_ingredient_matrix = cuisine_ingredient_matrix.div(cuisine_ingredient_matrix.sum(axis=1), axis=0)
        
        # Create heatmap with improved styling
        fig = go.Figure(data=go.Heatmap(
            z=cuisine_ingredient_matrix.values,
            x=cuisine_ingredient_matrix.columns,
            y=cuisine_ingredient_matrix.index,
            colorscale=[
                [0, 'rgb(255,255,255)'],
                [0.2, 'rgb(255,237,222)'],
                [0.4, 'rgb(255,190,142)'],
                [0.6, 'rgb(255,117,71)'],
                [0.8, 'rgb(225,56,36)'],
                [1, 'rgb(170,0,0)']
            ],
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         'Ingredient: <b>%{x}</b><br>' +
                         'Usage: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Ingredient Usage Patterns Across Cuisines',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            width=1200,
            height=800,
            xaxis={
                'title': 'Ingredients',
                'tickangle': 45,
                'side': 'bottom',
                'tickfont': {'size': 12}
            },
            yaxis={
                'title': 'Cuisines',
                'tickfont': {'size': 12}
            },
            margin=dict(l=150, r=20, t=80, b=150)
        )
        
        # Save the visualization
        fig.write_html(vis_dir / 'ingredient_heatmap.html')
        logging.info("Ingredients heatmap visualization completed")
        
    except Exception as e:
        logging.error(f"Error in ingredients heatmap visualization: {str(e)}")
        raise


    
# def create_interactive_cooking_dendrogram(df, vis_dir):
#     """Create a fully interactive dendrogram with hoverable lines and points."""
#     try:
#         logging.info("Starting cooking dendrogram visualization...")
        
#         # Data preparation
#         df['cooking_method'] = df['cooking_method'].fillna('unknown')
#         df['cooking_method'] = df['cooking_method'].str.strip().str.lower()
#         cuisine_method = pd.crosstab(df['cuisine'], df['cooking_method'])
        
#         # Filter for major cuisines
#         min_recipes = 3
#         cuisine_counts = df['cuisine'].value_counts()
#         major_cuisines = cuisine_counts[cuisine_counts >= min_recipes].index
#         cuisine_method = cuisine_method.loc[major_cuisines]
        
#         # Calculate percentages
#         cuisine_method_pct = cuisine_method.div(cuisine_method.sum(axis=1), axis=0) * 100
        
#         # Calculate linkage
#         Z = linkage(cuisine_method_pct, method='ward', optimal_ordering=True)
        
#         # Define clusters
#         n_clusters = 5
#         clusters = fcluster(Z, n_clusters, criterion='maxclust')
        
#         # Define colors for clusters
#         colors = {
#             1: '#1f77b4',  # blue
#             2: '#ff7f0e',  # orange
#             3: '#2ca02c',  # green
#             4: '#d62728',  # red
#             5: '#9467bd'   # purple
#         }
        
#         # Get dendrogram data
#         dn = dendrogram(Z, labels=cuisine_method_pct.index, no_plot=True)
        
#         # Create figure
#         fig = go.Figure()
        
#         # Calculate x positions with better spacing
#         x_scale = 100
#         x_positions = np.arange(len(dn['ivl'])) * x_scale
        
#         # Create a mapping of cuisine to cluster
#         cuisine_to_cluster = dict(zip(cuisine_method_pct.index, clusters))
#         cuisine_to_methods = {}
        
#         # Prepare hover text for each cuisine
#         for cuisine in cuisine_method_pct.index:
#             methods = cuisine_method_pct.loc[cuisine].sort_values(ascending=False)
#             top_methods = methods[methods > 0].head(3)
#             cuisine_to_methods[cuisine] = "<br>".join([f"- {m}: {v:.1f}%" for m, v in top_methods.items()])
        
#         # Add dendrogram traces with enhanced hover information
#         for i, (xi, yi) in enumerate(zip(dn['icoord'], dn['dcoord'])):
#             # Scale x coordinates
#             xi_scaled = [x * x_scale / 10 for x in xi]
            
#             # Determine connected cuisines
#             left_idx = int(xi[0] / 10)
#             right_idx = int(xi[-1] / 10)
#             if 0 <= left_idx < len(dn['ivl']) and 0 <= right_idx < len(dn['ivl']):
#                 left_cuisine = dn['ivl'][left_idx]
#                 right_cuisine = dn['ivl'][right_idx]
                
#                 # Get cluster information
#                 left_cluster = cuisine_to_cluster[left_cuisine]
#                 right_cluster = cuisine_to_cluster[right_cuisine]
                
#                 # Use color of the lower cluster number
#                 cluster_num = min(left_cluster, right_cluster)
#                 color = colors[cluster_num]
                
#                 # Create hover text showing connection information
#                 hover_text = (
#                     f"Connection between:<br>"
#                     f"<b>{left_cuisine}</b> (Cluster {left_cluster})<br>"
#                     f"{cuisine_to_methods[left_cuisine]}<br><br>"
#                     f"<b>{right_cuisine}</b> (Cluster {right_cluster})<br>"
#                     f"{cuisine_to_methods[right_cuisine]}"
#                 )
#             else:
#                 color = 'rgba(128, 128, 128, 0.5)'
#                 hover_text = "Internal connection"
            
#             # Add trace for the connection line
#             fig.add_trace(go.Scatter(
#                 x=xi_scaled,
#                 y=yi,
#                 mode='lines',
#                 line=dict(
#                     color=color,
#                     width=2
#                 ),
#                 hoverinfo='text',
#                 hovertext=hover_text,
#                 showlegend=False
#             ))
        
#         # Add cuisine points with enhanced hover information
#         for i, label in enumerate(dn['ivl']):
#             cluster_num = cuisine_to_cluster[label]
            
#             hover_text = (
#                 f"<b>{label}</b> (Cluster {cluster_num})<br><br>"
#                 f"Cooking methods:<br>{cuisine_to_methods[label]}"
#             )
            
#             fig.add_trace(go.Scatter(
#                 x=[x_positions[i]],
#                 y=[0],
#                 mode='markers+text',
#                 marker=dict(
#                     size=10,
#                     color=colors[cluster_num],
#                     line=dict(color='white', width=1)
#                 ),
#                 text=label,
#                 textposition='bottom center',
#                 hovertext=hover_text,
#                 hoverinfo='text',
#                 showlegend=False
#             ))
        
#         # Add cluster information boxes
#         for cluster in range(1, n_clusters + 1):
#             cluster_cuisines = [c for c, cl in cuisine_to_cluster.items() if cl == cluster]
            
#             # Calculate average methods for cluster
#             cluster_methods = cuisine_method_pct.loc[cluster_cuisines].mean()
#             top_methods = cluster_methods.nlargest(3)
            
#             info_text = (
#                 f"<b>Cluster {cluster}</b> ({len(cluster_cuisines)} cuisines)<br>"
#                 f"Common cooking methods:<br>"
#                 + "<br>".join([f"- {m}: {v:.1f}%" for m, v in top_methods.items()])
#             )
            
#             fig.add_annotation(
#                 x=1.15,
#                 y=0.95 - (cluster-1) * 0.18,
#                 xref="paper",
#                 yref="paper",
#                 text=info_text,
#                 showarrow=False,
#                 font=dict(size=12),
#                 align="left",
#                 bgcolor='rgba(255,255,255,0.95)',
#                 bordercolor=colors[cluster],
#                 borderwidth=2,
#                 borderpad=4
#             )
        
#         # Update layout
#         fig.update_layout(
#             title={
#                 'text': 'Cuisine Clustering by Cooking Methods',
#                 'y': 0.98,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top',
#                 'font': dict(size=24)
#             },
#             width=1200,
#             height=800,
#             showlegend=False,
#             xaxis={
#                 'showgrid': False,
#                 'zeroline': False,
#                 'showline': False,
#                 'showticklabels': False,
#                 'range': [-x_scale, max(x_positions) + x_scale]
#             },
#             yaxis={
#                 'title': 'Distance (Dissimilarity)',
#                 'showgrid': True,
#                 'gridcolor': 'rgba(128,128,128,0.2)',
#                 'zeroline': False
#             },
#             plot_bgcolor='white',
#             paper_bgcolor='white',
#             margin=dict(l=100, r=300, t=100, b=100),
#             hovermode='closest'
#         )
        
#         # Add subtitle
#         fig.add_annotation(
#             text='Hover over lines or cuisines to see relationships and cooking methods',
#             xref="paper",
#             yref="paper",
#             x=0.5,
#             y=1.05,
#             showarrow=False,
#             font=dict(size=14, color="gray"),
#         )
        
#         # Save the visualization
#         fig.write_html(vis_dir / 'cooking_dendrogram.html')
#         logging.info("Cooking dendrogram visualization completed")
        
#     except Exception as e:
#         logging.error(f"Error in cooking dendrogram visualization: {str(e)}")
#         raise

def create_interactive_cooking_dendrogram(df, vis_dir):
    """Create a dendrogram with consistent colors and better box positioning."""
    try:
        logging.info("Starting cooking dendrogram visualization...")
        
        # Imports
        import plotly.graph_objects as go
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        import numpy as np
        
        # Data preparation
        df['cooking_method'] = df['cooking_method'].fillna('unknown')
        df['cooking_method'] = df['cooking_method'].str.strip().str.lower()
        cuisine_method = pd.crosstab(df['cuisine'], df['cooking_method'])
        
        # Filter for major cuisines
        min_recipes = 3
        cuisine_counts = df['cuisine'].value_counts()
        major_cuisines = cuisine_counts[cuisine_counts >= min_recipes].index
        cuisine_method = cuisine_method.loc[major_cuisines]
        
        # Calculate percentages
        cuisine_method_pct = cuisine_method.div(cuisine_method.sum(axis=1), axis=0) * 100
        
        # Calculate linkage
        Z = linkage(cuisine_method_pct, method='ward', optimal_ordering=True)
        
        # Define number of clusters and get cluster assignments
        n_clusters = 5  # Define this here
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Define colors to match the visualization exactly
        colors = {
            1: 'rgb(0, 114, 178)',      # blue for cluster 1
            2: 'rgb(255, 127, 0)',      # orange for cluster 2
            3: 'rgb(34, 139, 34)',      # green for cluster 3
            4: 'rgb(220, 20, 60)',      # red for cluster 4
            5: 'rgb(148, 0, 211)'       # purple for cluster 5
        }
        
        # Get dendrogram data
        dn = dendrogram(Z, labels=cuisine_method_pct.index, no_plot=True)
        
        # Initialize plotly figure
        fig = go.Figure()
        
        # Calculate x positions
        x_scale = 100
        x_positions = np.arange(len(dn['ivl'])) * x_scale
        
        # Create cuisine to cluster mapping
        cuisine_to_cluster = dict(zip(cuisine_method_pct.index, clusters))
        cuisine_to_methods = {}
        
        # Prepare hover text for each cuisine
        for cuisine in cuisine_method_pct.index:
            methods = cuisine_method_pct.loc[cuisine].sort_values(ascending=False)
            top_methods = methods[methods > 0].head(3)
            cuisine_to_methods[cuisine] = "<br>".join([f"- {m}: {v:.1f}%" for m, v in top_methods.items()])
        
        # Add dendrogram traces
        for i, (xi, yi) in enumerate(zip(dn['icoord'], dn['dcoord'])):
            xi_scaled = [x * x_scale / 10 for x in xi]
            
            left_idx = int(xi[0] / 10)
            right_idx = int(xi[-1] / 10)
            
            if 0 <= left_idx < len(dn['ivl']) and 0 <= right_idx < len(dn['ivl']):
                left_cuisine = dn['ivl'][left_idx]
                right_cuisine = dn['ivl'][right_idx]
                cluster_color = colors[min(cuisine_to_cluster[left_cuisine], 
                                        cuisine_to_cluster[right_cuisine])]
                
                hover_text = (
                    f"Connection between:<br>"
                    f"<b>{left_cuisine}</b> (Cluster {cuisine_to_cluster[left_cuisine]})<br>"
                    f"{cuisine_to_methods[left_cuisine]}<br><br>"
                    f"<b>{right_cuisine}</b> (Cluster {cuisine_to_cluster[right_cuisine]})<br>"
                    f"{cuisine_to_methods[right_cuisine]}"
                )
            else:
                cluster_color = 'rgba(128, 128, 128, 0.5)'
                hover_text = "Internal connection"
            
            fig.add_trace(
                go.Scatter(
                    x=xi_scaled,
                    y=yi,
                    mode='lines',
                    line=dict(color=cluster_color, width=2),
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=False
                )
            )
        
        # Add cuisine points
        for i, cuisine in enumerate(dn['ivl']):
            cluster_num = cuisine_to_cluster[cuisine]
            
            hover_text = (
                f"<b>{cuisine}</b> (Cluster {cluster_num})<br><br>"
                f"Cooking methods:<br>{cuisine_to_methods[cuisine]}"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[i]],
                    y=[0],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=colors[cluster_num],
                        line=dict(color='white', width=1)
                    ),
                    text=cuisine,
                    textposition='bottom center',
                    hovertext=hover_text,
                    hoverinfo='text',
                    showlegend=False
                )
            )
        
        # Define y-positions for cluster information boxes
        y_positions = {
            1: 0.95,    # Top position
            2: 0.75,    # More space between boxes
            3: 0.55,    # More space between boxes
            4: 0.35,    # More space between boxes
            5: 0.15     # Bottom position
        }
        
        # Add cluster information boxes with better spacing
        for cluster in range(1, n_clusters + 1):
            cluster_cuisines = [c for c, cl in cuisine_to_cluster.items() if cl == cluster]
            
            cluster_methods = cuisine_method_pct.loc[cluster_cuisines].mean()
            top_methods = cluster_methods.nlargest(3)
            
            info_text = (
                f"<b>Cluster {cluster}</b> ({len(cluster_cuisines)} cuisines)<br>"
                f"Common cooking methods:<br>"
                + "<br>".join([f"- {m}: {v:.1f}%" for m, v in top_methods.items()])
            )
            
            fig.add_annotation(
                x=1.25,
                y=y_positions[cluster],
                xref="paper",
                yref="paper",
                text=info_text,
                showarrow=False,
                font=dict(size=12),
                align="left",
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=colors[cluster],
                borderwidth=2,
                borderpad=4
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Cuisine Clustering by Cooking Methods',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            width=1400,
            height=900,
            showlegend=False,
            xaxis={
                'showgrid': False,
                'zeroline': False,
                'showline': False,
                'showticklabels': False,
                'range': [-x_scale, max(x_positions) + x_scale]
            },
            yaxis={
                'title': 'Distance (Dissimilarity)',
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'zeroline': False
            },
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(
                l=100,
                r=450,
                t=100,
                b=100,
                pad=4
            ),
            hovermode='closest'
        )
        
        # Add subtitle
        fig.add_annotation(
            text='Hover over lines or cuisines to see relationships and cooking methods',
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        
        # Save the visualization
        fig.write_html(vis_dir / 'cooking_dendrogram.html')
        logging.info("Cooking dendrogram visualization completed")
        
    except Exception as e:
        logging.error(f"Error in cooking dendrogram visualization: {str(e)}")
        raise
    
def generate_visualizations(df, similarity_matrix, clusters):
    """Generate all interactive visualizations."""
    try:
        # Create visualizations directory
        vis_dir = Path('visualizations')
        vis_dir.mkdir(exist_ok=True)
        
        logging.info("Starting visualization generation...")
        
        # Create visualizations
        create_interactive_cuisine_network(df, similarity_matrix, vis_dir)
        create_interactive_ingredients_heatmap(df, vis_dir)
        create_interactive_cooking_dendrogram(df, vis_dir)
        
        # Create enhanced HTML index
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Recipe Analysis</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: auto;
                }
                .viz-container {
                    margin-bottom: 40px;
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .viz-header {
                    margin-bottom: 20px;
                }
                .viz-title {
                    font-size: 24px;
                    font-weight: 600;
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }
                .viz-description {
                    color: #666;
                    margin: 0 0 20px 0;
                    font-size: 16px;
                }
                iframe {
                    width: 100%;
                    height: 800px;
                    border: none;
                    border-radius: 8px;
                    background: #fff;
                }
                h1 {
                    text-align: center;
                    color: #2c3e50;
                    margin: 40px 0;
                    font-size: 36px;
                    font-weight: 700;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Interactive Recipe Analysis Dashboard</h1>
                
                <div class="viz-container">
                    <div class="viz-header">
                        <h2 class="viz-title">Cuisine Network Analysis</h2>
                        <p class="viz-description">
                            Explore relationships between different cuisines based on shared ingredients and cooking methods.
                            Larger nodes indicate more recipes, and thicker connections show stronger similarities.
                            Hover over nodes and edges for detailed information.
                        </p>
                    </div>
                    <iframe src="cuisine_network.html"></iframe>
                </div>
                
                <div class="viz-container">
                    <div class="viz-header">
                        <h2 class="viz-title">Ingredient Usage Patterns</h2>
                        <p class="viz-description">
                            Discover how ingredients are used across different cuisines.
                            Darker colors indicate higher usage frequency.
                            Hover over cells to see exact usage percentages.
                        </p>
                    </div>
                    <iframe src="ingredient_heatmap.html"></iframe>
                </div>
                
                <div class="viz-container">
                    <div class="viz-header">
                        <h2 class="viz-title">Cuisine Clustering Analysis</h2>
                        <p class="viz-description">
                            See how cuisines are grouped based on their cooking methods.
                            Colors indicate different clusters of similar cooking styles.
                            The closer cuisines are connected, the more similar their cooking approaches.
                        </p>
                    </div>
                    <iframe src="cooking_dendrogram.html"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(vis_dir / 'index.html', 'w') as f:
            f.write(html_content)
        
        logging.info("All visualizations generated successfully")
        return vis_dir
        
    except Exception as e:
        logging.error(f"Error in visualization generation: {str(e)}")
        raise