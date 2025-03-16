import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import gc
import pickle
import os

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the model class (same as in training)
class BipartiteCitationGNN(torch.nn.Module):
    def __init__(self, author_in_channels, hidden_channels, out_channels, num_papers, dropout=0.2):
        super().__init__()
        
        # Embedding layer for papers
        self.paper_embedding = torch.nn.Embedding(num_papers, hidden_channels)
        
        # Author feature projection
        self.author_proj = Linear(author_in_channels, hidden_channels)
        
        # Dropout for regularization
        self.dropout = dropout
        
        # Heterogeneous graph convolution
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'written_by', 'author'): SAGEConv((-1, -1), hidden_channels),
        })
        
        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
            ('paper', 'written_by', 'author'): SAGEConv((-1, -1), hidden_channels),
        })
        
        # Prediction layers for papers
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        # Process node features
        x_dict = {
            'author': self.author_proj(x_dict['author']),
            'paper': self.paper_embedding.weight,
        }
        
        # Apply dropout for regularization
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                 for key, x in x_dict.items()}
        
        # First message passing layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                 for key, x in x_dict.items()}
        
        # Second message passing layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Prediction for papers
        paper_x = x_dict['paper']
        paper_x = F.dropout(paper_x, p=self.dropout, training=self.training)
        paper_x = F.relu(self.lin1(paper_x))
        paper_x = self.lin2(paper_x)
        
        return paper_x

def main():
    print("Starting data loading...")
    start_time = time.time()

    # Load the data for testing
    authors = pd.read_csv('data/PhilosophyAuthors_2000_16_Profile.csv')
    test_papers = pd.read_csv('data/PhilosophyPapers_2016_test.csv')
    test_edges = pd.read_csv('data/Philosophy_2016_edges_test.csv')

    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    print(f"Authors shape: {authors.shape}")
    print(f"Test papers shape: {test_papers.shape}")
    print(f"Test edges shape: {test_edges.shape}")

    # Load the trained model and mappings
    model_path = 'models/citation_gnn_model.pt'
    mappings_path = 'models/id_mappings.pkl'

    if not os.path.exists(model_path) or not os.path.exists(mappings_path):
        print("Error: Model or ID mappings not found. Please run train_gnn.py first.")
        return

    # Load the checkpoint with weights_only=False to handle the pickle error
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load the mappings
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    author_id_map = mappings['author_id_map']
    paper_id_map = mappings['paper_id_map']
    
    # Extract model hyperparameters
    author_in_channels = checkpoint['author_in_channels']
    hidden_channels = checkpoint['hidden_channels']
    out_channels = checkpoint['out_channels']
    num_papers_train = checkpoint['num_papers']

    print("\nFiltering data for test...")
    start_time = time.time()

    # Get the list of authors we have features for
    available_authors = set(authors['AuthorID'].values)
    
    # Filter test edges to only include authors with features
    test_edges = test_edges[test_edges['AuthorID'].isin(available_authors)]
    print(f"Test edges after author filtering: {test_edges.shape}")
    
    # Process new papers that weren't seen during training
    available_test_papers = set(test_papers['PaperID'].values)
    test_edges = test_edges[test_edges['PaperID'].isin(available_test_papers)]
    print(f"Test edges after paper filtering: {test_edges.shape}")
    
    # Create mappings for new papers not seen during training
    # Get papers and authors in the test set
    test_paper_ids = set(test_papers['PaperID'].values)
    test_author_ids = set(test_edges['AuthorID'].values)
    
    # Find new papers (not in training set)
    new_paper_ids = test_paper_ids - set(paper_id_map.keys())
    print(f"Number of new papers in test set: {len(new_paper_ids)}")

    # Extend paper_id_map for new papers
    max_paper_id = max(paper_id_map.values())
    for i, paper_id in enumerate(new_paper_ids):
        paper_id_map[paper_id] = max_paper_id + i + 1
    
    # Update num_papers to include new papers
    num_papers = max(paper_id_map.values()) + 1
    print(f"Total number of papers after adding test papers: {num_papers}")
    
    print(f"Data filtering completed in {time.time() - start_time:.2f} seconds")

    print("\nProcessing author features...")
    start_time = time.time()

    # Filter the authors dataframe
    authors = authors[authors['AuthorID'].isin(author_id_map.keys())].copy()
    
    # Fill NaN values with 0
    authors = authors.fillna(0)
    
    # Create a new index column based on the mapping
    authors['new_idx'] = authors['AuthorID'].map(author_id_map)
    
    # Sort by the new index to ensure correct order
    authors = authors.sort_values(by='new_idx')
    
    # Drop the AuthorID and new_idx columns to get just the features
    author_features_df = authors.drop(['AuthorID', 'new_idx'], axis=1)
    
    # Convert to tensor
    author_features = torch.tensor(author_features_df.values, dtype=torch.float)
    print(f"Author features shape: {author_features.shape}")
    
    print(f"Author feature processing completed in {time.time() - start_time:.2f} seconds")

    # Free up memory
    del authors, author_features_df
    gc.collect()

    print("\nCreating edge indices for test data...")
    start_time = time.time()

    # Create edge indices for test data
    test_author_indices = torch.tensor([author_id_map[author_id] for author_id in test_edges['AuthorID'].values 
                                        if author_id in author_id_map], dtype=torch.long)
    test_paper_indices_edges = torch.tensor([paper_id_map[paper_id] for paper_id in test_edges['PaperID'].values 
                                            if paper_id in paper_id_map], dtype=torch.long)
    
    # Make sure the arrays are the same length after filtering
    min_len = min(len(test_author_indices), len(test_paper_indices_edges))
    test_author_indices = test_author_indices[:min_len]
    test_paper_indices_edges = test_paper_indices_edges[:min_len]
    
    test_edge_index = torch.stack([test_author_indices, test_paper_indices_edges])
    print(f"Test edge index shape: {test_edge_index.shape}")
    
    # Free up memory
    del test_edges, test_author_indices, test_paper_indices_edges
    gc.collect()

    print("\nPreparing target values for test data...")
    start_time = time.time()
    
    # Filter test papers to only include those with mappings
    test_papers_filtered = test_papers[test_papers['PaperID'].isin(paper_id_map.keys())].copy()
    print(f"Test papers after filtering: {test_papers_filtered.shape}")
    
    # Add new indices to the papers dataframe
    test_papers_filtered['new_idx'] = test_papers_filtered['PaperID'].map(paper_id_map)
    
    # Remove papers that don't have a mapping
    test_papers_filtered = test_papers_filtered.dropna(subset=['new_idx'])
    
    # Map to the new indices
    test_paper_indices = torch.tensor(test_papers_filtered['new_idx'].values, dtype=torch.long)
    test_y = torch.tensor(test_papers_filtered['C5_log'].values, dtype=torch.float)
    
    print(f"Test paper indices shape: {test_paper_indices.shape}")
    print(f"Test target shape: {test_y.shape}")
    
    # Original IDs for later mapping back
    original_paper_ids = test_papers_filtered['PaperID'].values
    
    print(f"Target value preparation completed in {time.time() - start_time:.2f} seconds")

    # Free up memory
    del test_papers
    gc.collect()

    print("\nCreating heterogeneous data object for test...")
    start_time = time.time()
    
    # Create edge index dictionary for model
    edge_index_dict = {
        ('author', 'writes', 'paper'): test_edge_index,
        ('paper', 'written_by', 'author'): test_edge_index.flip([0]),
    }
    
    # Create node feature dictionary
    x_dict = {
        'author': author_features,
    }
    
    print(f"Heterogeneous data creation completed in {time.time() - start_time:.2f} seconds")
    
    print("\nInitializing and loading model...")
    # Initialize the model with the original num_papers from training
    num_papers_train = checkpoint['num_papers']
    model = BipartiteCitationGNN(author_in_channels, hidden_channels, out_channels, num_papers_train)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Now we need to extend the paper embeddings to handle new papers
    extended_embedding = torch.nn.Embedding(num_papers, hidden_channels)
    # Copy the trained paper embeddings
    with torch.no_grad():
        extended_embedding.weight[:num_papers_train] = model.paper_embedding.weight
        # Initialize new paper embeddings with random values
        if num_papers > num_papers_train:
            torch.nn.init.xavier_uniform_(extended_embedding.weight[num_papers_train:])
    
    # Replace the embedding layer
    model.paper_embedding = extended_embedding
    
    model = model.to(device)
    model.eval()
    
    # Move data to device
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    test_paper_indices = test_paper_indices.to(device)
    test_y = test_y.to(device)
    
    print("Making predictions...")
    # Make predictions
    with torch.no_grad():
        paper_pred = model(x_dict, edge_index_dict)
        test_pred = paper_pred[test_paper_indices].squeeze()
        
        # Calculate test metrics
        test_loss = F.mse_loss(test_pred, test_y)
        
        # Move tensors to CPU for numpy operations
        test_rmse = mean_squared_error(test_y.cpu().numpy(), test_pred.cpu().numpy(), squared=False)
        test_mae = mean_absolute_error(test_y.cpu().numpy(), test_pred.cpu().numpy())
        
        print(f"Test Loss: {test_loss.item():.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Create dataframe for test predictions
        test_predictions = pd.DataFrame({
            'PaperID': original_paper_ids,
            'Actual_C5_log': test_y.cpu().numpy(),
            'Predicted_C5_log': test_pred.cpu().numpy()
        })
        
        # Save predictions
        test_predictions.to_csv('citation_predictions.csv', index=False)
        print("Predictions saved to citation_predictions.csv")

if __name__ == "__main__":
    main()