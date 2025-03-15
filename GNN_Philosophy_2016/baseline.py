import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data
authors = pd.read_csv('data/PhilosophyAuthors_2000_16_Profile.csv')
train_papers = pd.read_csv('data/PhilosophyPapers_2000_15.csv')
train_edges = pd.read_csv('data/Philosophy_2000_15_edges.csv')
test_papers = pd.read_csv('data/PhilosophyPapers_2016.csv')
test_edges = pd.read_csv('data/Philosophy_2016_edges.csv')

print(f"Authors shape: {authors.shape}")
print(f"Train papers shape: {train_papers.shape}")
print(f"Train edges shape: {train_edges.shape}")
print(f"Test papers shape: {test_papers.shape}")
print(f"Test edges shape: {test_edges.shape}")

# First, filter edges to only include authors and papers in our datasets
# Get the list of authors we have features for
available_authors = set(authors['AuthorID'].values)
print(f"Number of authors with features: {len(available_authors)}")

# Filter edges to only include authors with features
train_edges = train_edges[train_edges['AuthorID'].isin(available_authors)]
test_edges = test_edges[test_edges['AuthorID'].isin(available_authors)]

print(f"Train edges after author filtering: {train_edges.shape}")
print(f"Test edges after author filtering: {test_edges.shape}")

# Filter to only include papers in our paper datasets
available_train_papers = set(train_papers['PaperID'].values)
available_test_papers = set(test_papers['PaperID'].values)
all_available_papers = available_train_papers.union(available_test_papers)

print(f"Number of available papers: {len(all_available_papers)}")

train_edges = train_edges[train_edges['PaperID'].isin(all_available_papers)]
test_edges = test_edges[test_edges['PaperID'].isin(all_available_papers)]

print(f"Train edges after paper filtering: {train_edges.shape}")
print(f"Test edges after paper filtering: {test_edges.shape}")

# Create unique IDs for filtered authors and papers
filtered_author_ids = pd.concat([train_edges['AuthorID'], test_edges['AuthorID']]).unique()
filtered_paper_ids = pd.concat([
    train_papers['PaperID'][train_papers['PaperID'].isin(set(train_edges['PaperID']))], 
    test_papers['PaperID'][test_papers['PaperID'].isin(set(test_edges['PaperID']))]
]).unique()

print(f"Number of unique authors after filtering: {len(filtered_author_ids)}")
print(f"Number of unique papers after filtering: {len(filtered_paper_ids)}")

# Create mapping dictionaries for authors and papers to ensure contiguous IDs
author_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_author_ids)}
paper_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_paper_ids)}

# Create author features
# Filter authors dataframe to only include authors in our filtered set
authors = authors[authors['AuthorID'].isin(filtered_author_ids)]
print(f"Authors shape after filtering: {authors.shape}")

# Fill NaN values with 0 or mean depending on the feature
authors = authors.fillna(0)

# Create a new dataframe that has the same order as our author_id_map
ordered_authors = pd.DataFrame(index=range(len(author_id_map)))
for old_id, new_id in author_id_map.items():
    if old_id in set(authors['AuthorID']):
        author_row = authors[authors['AuthorID'] == old_id].iloc[0].drop('AuthorID')
        for col in author_row.index:
            ordered_authors.loc[new_id, col] = author_row[col]

ordered_authors = ordered_authors.fillna(0)  # Fill any missing values
author_features = torch.tensor(ordered_authors.values, dtype=torch.float)
print(f"Author features shape: {author_features.shape}")

# For papers, we don't have explicit features, so we'll use an embedding layer
# But we need to know how many papers we have
num_papers = len(paper_id_map)
print(f"Number of papers for embedding: {num_papers}")

# Create edge indices for the bipartite graph
# Training edges
train_edge_index = torch.tensor([
    [author_id_map[author_id] for author_id in train_edges['AuthorID'].values],
    [paper_id_map[paper_id] for paper_id in train_edges['PaperID'].values]
], dtype=torch.long)

# Test edges
test_edge_index = torch.tensor([
    [author_id_map[author_id] for author_id in test_edges['AuthorID'].values],
    [paper_id_map[paper_id] for paper_id in test_edges['PaperID'].values]
], dtype=torch.long)

print(f"Train edge index shape: {train_edge_index.shape}")
print(f"Test edge index shape: {test_edge_index.shape}")

# Combine all edges for the model
combined_edge_index = torch.cat([train_edge_index, test_edge_index], dim=1)
print(f"Combined edge index shape: {combined_edge_index.shape}")

# Verify index ranges
print(f"Max author index in edge index: {combined_edge_index[0].max().item()}")
print(f"Max paper index in edge index: {combined_edge_index[1].max().item()}")
print(f"Author features size: {author_features.size(0)}")
print(f"Number of papers: {num_papers}")

# Create target values for training and testing
# Filter papers to only include those in our graph
train_papers_filtered = train_papers[train_papers['PaperID'].isin(filtered_paper_ids)]
test_papers_filtered = test_papers[test_papers['PaperID'].isin(filtered_paper_ids)]

print(f"Train papers after filtering: {train_papers_filtered.shape}")
print(f"Test papers after filtering: {test_papers_filtered.shape}")

# Map to the new indices
train_paper_indices = torch.tensor([
    paper_id_map[pid] for pid in train_papers_filtered['PaperID'].values if pid in paper_id_map
], dtype=torch.long)
test_paper_indices = torch.tensor([
    paper_id_map[pid] for pid in test_papers_filtered['PaperID'].values if pid in paper_id_map
], dtype=torch.long)

train_y = torch.tensor(
    [train_papers_filtered['C5'].values[i] for i, pid in enumerate(train_papers_filtered['PaperID'].values) 
     if pid in paper_id_map], dtype=torch.float
)
test_y = torch.tensor(
    [test_papers_filtered['C5'].values[i] for i, pid in enumerate(test_papers_filtered['PaperID'].values)
     if pid in paper_id_map], dtype=torch.float
)

print(f"Train paper indices shape: {train_paper_indices.shape}")
print(f"Test paper indices shape: {test_paper_indices.shape}")
print(f"Train target shape: {train_y.shape}")
print(f"Test target shape: {test_y.shape}")

# Create heterogeneous data objects for PyTorch Geometric
data = HeteroData()

# Add node types
data['author'].x = author_features
data['paper'].num_nodes = num_papers  # We'll use an embedding layer for papers

# Add edge types: author-writes->paper and paper-written_by->author
data['author', 'writes', 'paper'].edge_index = combined_edge_index
data['paper', 'written_by', 'author'].edge_index = combined_edge_index.flip([0])

# Define the model
class BipartiteCitationGNN(torch.nn.Module):
    def __init__(self, author_in_channels, hidden_channels, out_channels, num_papers):
        super().__init__()
        
        # Embedding layer for papers
        self.paper_embedding = torch.nn.Embedding(num_papers, hidden_channels)
        
        # Author feature projection
        self.author_proj = Linear(author_in_channels, hidden_channels)
        
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
        
        # First message passing layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second message passing layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Prediction for papers
        paper_x = x_dict['paper']
        paper_x = F.relu(self.lin1(paper_x))
        paper_x = self.lin2(paper_x)
        
        return paper_x

# Create edge index dictionary
edge_index_dict = {
    ('author', 'writes', 'paper'): combined_edge_index,
    ('paper', 'written_by', 'author'): combined_edge_index.flip([0]),
}

# Create node feature dictionary
x_dict = {
    'author': author_features,
}

# Initialize the model
author_in_channels = author_features.size(1)
hidden_channels = 64
out_channels = 1
model = BipartiteCitationGNN(author_in_channels, hidden_channels, out_channels, num_papers)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    paper_pred = model(x_dict, edge_index_dict)
    
    # Calculate loss on training papers
    loss = F.mse_loss(paper_pred[train_paper_indices].squeeze(), train_y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Testing function
def test():
    model.eval()
    with torch.no_grad():
        paper_pred = model(x_dict, edge_index_dict)
        train_pred = paper_pred[train_paper_indices].squeeze()
        test_pred = paper_pred[test_paper_indices].squeeze()
        
        train_loss = F.mse_loss(train_pred, train_y)
        test_loss = F.mse_loss(test_pred, test_y)
        
        train_rmse = mean_squared_error(train_y.numpy(), train_pred.numpy(), squared=False)
        test_rmse = mean_squared_error(test_y.numpy(), test_pred.numpy(), squared=False)
        
        train_mae = mean_absolute_error(train_y.numpy(), train_pred.numpy())
        test_mae = mean_absolute_error(test_y.numpy(), test_pred.numpy())
        
    return train_loss.item(), test_loss.item(), train_rmse, test_rmse, train_mae, test_mae

# Train the model
epochs = 200
best_test_rmse = float('inf')
for epoch in range(1, epochs + 1):
    loss = train()
    
    # Evaluate every 10 epochs
    if epoch % 10 == 0:
        train_loss, test_loss, train_rmse, test_rmse, train_mae, test_mae = test()
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        print(f'Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
        print(f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}')
        
        # Save best model
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            torch.save(model.state_dict(), 'best_citation_gnn.pt')

# Load the best model and make predictions
model.load_state_dict(torch.load('best_citation_gnn.pt'))
model.eval()

with torch.no_grad():
    paper_pred = model(x_dict, edge_index_dict)
    test_pred = paper_pred[test_paper_indices].squeeze().numpy()

# Create dataframe for test predictions
test_predictions = pd.DataFrame({
    'PaperID': [filtered_paper_ids[i] for i in test_paper_indices.numpy()],
    'Actual_C5': test_y.numpy(),
    'Predicted_C5': test_pred
})

# Save predictions
test_predictions.to_csv('citation_predictions.csv', index=False)
print("Predictions saved to citation_predictions.csv")