import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import gc
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Starting data loading...")
start_time = time.time()

# Load the data
authors = pd.read_csv('data/Philosophy_2016_Author_Profiles.csv')
train_papers = pd.read_csv('data/PhilosophyPapers_2016_train.csv')
train_edges = pd.read_csv('data/Philosophy_2016_edges_train.csv')
val_papers = pd.read_csv('data/PhilosophyPapers_2016_validation.csv')
val_edges = pd.read_csv('data/Philosophy_2016_edges_validation.csv')

print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
print(f"Authors shape: {authors.shape}")
print(f"Train papers shape: {train_papers.shape}")
print(f"Train edges shape: {train_edges.shape}")
print(f"Validation papers shape: {val_papers.shape}")
print(f"Validation edges shape: {val_edges.shape}")

print("\nFiltering data...")
start_time = time.time()

# First, filter edges to only include authors and papers in our datasets
# Get the list of authors we have features for
available_authors = set(authors['AuthorID'].values)
print(f"Number of authors with features: {len(available_authors)}")

# Filter edges to only include authors with features
train_edges = train_edges[train_edges['AuthorID'].isin(available_authors)]
val_edges = val_edges[val_edges['AuthorID'].isin(available_authors)]

print(f"Train edges after author filtering: {train_edges.shape}")
print(f"Validation edges after author filtering: {val_edges.shape}")

# Filter to only include papers in our paper datasets
available_train_papers = set(train_papers['PaperID'].values)
available_val_papers = set(val_papers['PaperID'].values)
all_available_papers = available_train_papers.union(available_val_papers)

print(f"Number of available papers: {len(all_available_papers)}")

train_edges = train_edges[train_edges['PaperID'].isin(all_available_papers)]
val_edges = val_edges[val_edges['PaperID'].isin(all_available_papers)]

print(f"Train edges after paper filtering: {train_edges.shape}")
print(f"Validation edges after paper filtering: {val_edges.shape}")

# Create unique IDs for filtered authors and papers
filtered_author_ids = pd.concat([train_edges['AuthorID'], val_edges['AuthorID']]).unique()
filtered_paper_ids = pd.concat([
    train_papers['PaperID'][train_papers['PaperID'].isin(set(train_edges['PaperID']))], 
    val_papers['PaperID'][val_papers['PaperID'].isin(set(val_edges['PaperID']))]
]).unique()

print(f"Number of unique authors after filtering: {len(filtered_author_ids)}")
print(f"Number of unique papers after filtering: {len(filtered_paper_ids)}")

# Create mapping dictionaries for authors and papers to ensure contiguous IDs
author_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_author_ids)}
paper_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_paper_ids)}

print(f"Data filtering completed in {time.time() - start_time:.2f} seconds")

print("\nProcessing author features...")
start_time = time.time()

# Filter the authors dataframe
authors = authors[authors['AuthorID'].isin(filtered_author_ids)].copy()
print(f"Authors shape after filtering: {authors.shape}")

# Fill NaN values with 0
authors = authors.fillna(0)
print("Filled the NaN values with 0 for the author features")

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

# For papers, we don't have explicit features, so we'll use an embedding layer
# But we need to know how many papers we have
num_papers = len(paper_id_map)
print(f"Number of papers for embedding: {num_papers}")

print("\nCreating edge indices...")
start_time = time.time()

# Create edge indices more efficiently
# Training edges
train_author_indices = torch.tensor([author_id_map[author_id] for author_id in train_edges['AuthorID'].values], dtype=torch.long)
train_paper_indices_edges = torch.tensor([paper_id_map[paper_id] for paper_id in train_edges['PaperID'].values], dtype=torch.long)
train_edge_index = torch.stack([train_author_indices, train_paper_indices_edges])

# Validation edges
val_author_indices = torch.tensor([author_id_map[author_id] for author_id in val_edges['AuthorID'].values], dtype=torch.long)
val_paper_indices_edges = torch.tensor([paper_id_map[paper_id] for paper_id in val_edges['PaperID'].values], dtype=torch.long)
val_edge_index = torch.stack([val_author_indices, val_paper_indices_edges])

# Free up memory after creating edge indices
del train_edges, val_edges, train_author_indices, train_paper_indices_edges, val_author_indices, val_paper_indices_edges
gc.collect()

print(f"Train edge index shape: {train_edge_index.shape}")
print(f"Validation edge index shape: {val_edge_index.shape}")

# Combine all edges for the model
combined_edge_index = torch.cat([train_edge_index, val_edge_index], dim=1)
print(f"Combined edge index shape: {combined_edge_index.shape}")

# Verify index ranges
print(f"Max author index in edge index: {combined_edge_index[0].max().item()}")
print(f"Max paper index in edge index: {combined_edge_index[1].max().item()}")
print(f"Author features size: {author_features.size(0)}")
print(f"Number of papers: {num_papers}")

print(f"Edge index creation completed in {time.time() - start_time:.2f} seconds")

print("\nPreparing target values...")
start_time = time.time()

# Filter papers to only include those in our graph
train_papers_filtered = train_papers[train_papers['PaperID'].isin(filtered_paper_ids)]
val_papers_filtered = val_papers[val_papers['PaperID'].isin(filtered_paper_ids)]

print(f"Train papers after filtering: {train_papers_filtered.shape}")
print(f"Validation papers after filtering: {val_papers_filtered.shape}")

# Add new indices to the papers dataframes
train_papers_filtered['new_idx'] = train_papers_filtered['PaperID'].map(paper_id_map)
val_papers_filtered['new_idx'] = val_papers_filtered['PaperID'].map(paper_id_map)

# Remove papers that don't have a mapping (shouldn't be any)
train_papers_filtered = train_papers_filtered.dropna(subset=['new_idx'])
val_papers_filtered = val_papers_filtered.dropna(subset=['new_idx'])

# Map to the new indices
train_paper_indices = torch.tensor(train_papers_filtered['new_idx'].values, dtype=torch.long)
val_paper_indices = torch.tensor(val_papers_filtered['new_idx'].values, dtype=torch.long)

train_y = torch.tensor(train_papers_filtered['C5'].values, dtype=torch.float)
val_y = torch.tensor(val_papers_filtered['C5'].values, dtype=torch.float)

# Free up memory
del train_papers, val_papers, train_papers_filtered, val_papers_filtered
gc.collect()

print(f"Train paper indices shape: {train_paper_indices.shape}")
print(f"Validation paper indices shape: {val_paper_indices.shape}")
print(f"Train target shape: {train_y.shape}")
print(f"Validation target shape: {val_y.shape}")

print(f"Target value preparation completed in {time.time() - start_time:.2f} seconds")

print("\nCreating heterogeneous data object...")
start_time = time.time()

# Create heterogeneous data objects for PyTorch Geometric
data = HeteroData()

# Add node types
data['author'].x = author_features
data['paper'].num_nodes = num_papers  # We'll use an embedding layer for papers

# Add edge types: author-writes->paper and paper-written_by->author
data['author', 'writes', 'paper'].edge_index = combined_edge_index
data['paper', 'written_by', 'author'].edge_index = combined_edge_index.flip([0])

print(f"Heterogeneous data creation completed in {time.time() - start_time:.2f} seconds")

# Define the model
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
        
        # Ensure outputs are non-negative (during training for better learning)
        # But don't apply rounding during training (only in inference)
        if not self.training:
            paper_x = torch.clamp(paper_x, min=0.0)
            
        return paper_x

# Create dictionaries for model input
edge_index_dict = {
    ('author', 'writes', 'paper'): combined_edge_index,
    ('paper', 'written_by', 'author'): combined_edge_index.flip([0]),
}

x_dict = {
    'author': author_features,
}

print("\nInitializing model...")
# Initialize the model
author_in_channels = author_features.size(1)
hidden_channels = 64
out_channels = 1
model = BipartiteCitationGNN(author_in_channels, hidden_channels, out_channels, num_papers)
model = model.to(device)

# Move data to device
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
train_paper_indices = train_paper_indices.to(device)
val_paper_indices = val_paper_indices.to(device)
train_y = train_y.to(device)
val_y = val_y.to(device)

# Custom loss function for citation prediction
# This is a modified MSE loss that puts more penalty on underestimating citations
def citation_aware_loss(pred, target, alpha=1.2):
    # Standard MSE
    mse = F.mse_loss(pred, target, reduction='none')
    
    # Calculate directional penalties
    # Penalty for underestimation (pred < target) is higher
    under_pred = pred < target
    directional_penalty = torch.ones_like(mse)
    directional_penalty[under_pred] = alpha  # Increase penalty for underestimation
    
    # Apply directional penalty
    weighted_mse = mse * directional_penalty
    
    return weighted_mse.mean()

# Define optimizer with learning rate scheduler and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Testing function
def evaluate():
    model.eval()
    with torch.no_grad():
        paper_pred = model(x_dict, edge_index_dict)
        train_pred_raw = paper_pred[train_paper_indices].squeeze()
        val_pred_raw = paper_pred[val_paper_indices].squeeze()
        
        # Calculate loss using raw predictions
        train_loss = citation_aware_loss(train_pred_raw, train_y)
        val_loss = citation_aware_loss(val_pred_raw, val_y)
        
        # Apply non-negative and rounding for evaluation metrics
        train_pred = torch.round(torch.clamp(train_pred_raw, min=0.0))
        val_pred = torch.round(torch.clamp(val_pred_raw, min=0.0))
        
        # Move tensors to CPU for numpy operations
        train_rmse = mean_squared_error(train_y.cpu().numpy(), train_pred.cpu().numpy())
        val_rmse = mean_squared_error(val_y.cpu().numpy(), val_pred.cpu().numpy())
        
        train_mae = mean_absolute_error(train_y.cpu().numpy(), train_pred.cpu().numpy())
        val_mae = mean_absolute_error(val_y.cpu().numpy(), val_pred.cpu().numpy())
        
        # Also calculate raw metrics (before rounding) for comparison
        train_rmse_raw = mean_squared_error(train_y.cpu().numpy(), train_pred_raw.cpu().numpy())
        val_rmse_raw = mean_squared_error(val_y.cpu().numpy(), val_pred_raw.cpu().numpy())
        
    return train_loss.item(), val_loss.item(), train_rmse, val_rmse, train_mae, val_mae, train_rmse_raw, val_rmse_raw

# Create directory for model
os.makedirs('models', exist_ok=True)
model_save_path = 'models/citation_gnn_model.pt'

# Train the model with early stopping
print("\nStarting training...")
epochs = 200
patience = 20
best_val_rmse = float('inf')
no_improve = 0

# Create a mini-batch sampler for memory efficiency
batch_size = 524288  # Adjust based on your GPU memory
train_indices_list = train_paper_indices.split(batch_size)
total_batches = len(train_indices_list)

for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    
    # Mini-batch training
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch_indices in enumerate(train_indices_list):
        optimizer.zero_grad()
        
        # Forward pass
        paper_pred = model(x_dict, edge_index_dict)
        
        # Calculate loss on batch
        batch_y = train_y[batch_idx * batch_size: min((batch_idx + 1) * batch_size, len(train_y))]
        # Use custom citation-aware loss
        loss = citation_aware_loss(paper_pred[batch_indices].squeeze(), batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Print progress for long-running training
        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
            print(f'Epoch: {epoch:03d}, Batch: {batch_idx+1}/{total_batches}, '
                  f'Loss: {loss.item():.4f}, Time: {time.time() - epoch_start:.2f}s')
    
    # Average loss over all batches
    epoch_loss /= total_batches
    
    # Evaluate every 5 epochs
    if epoch % 5 == 0:
        eval_start = time.time()
        train_loss, val_loss, train_rmse, val_rmse, train_mae, val_mae, train_rmse_raw, val_rmse_raw = evaluate()
        
        print(f'Epoch: {epoch:03d}, Time: {time.time() - epoch_start:.2f}s, '
              f'Eval Time: {time.time() - eval_start:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train RMSE (rounded): {train_rmse:.4f}, Val RMSE (rounded): {val_rmse:.4f}')
        print(f'Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}')
        print(f'Train RMSE (raw): {train_rmse_raw:.4f}, Val RMSE (raw): {val_rmse_raw:.4f}')
        
        # Update learning rate based on validation performance
        scheduler.step(val_rmse)
        
        # Save best model and check for early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_rmse_raw': val_rmse_raw,
                'author_id_map': author_id_map,
                'paper_id_map': paper_id_map,
                'author_in_channels': author_in_channels,
                'hidden_channels': hidden_channels,
                'out_channels': out_channels,
                'num_papers': num_papers,
            }, model_save_path)
            
            no_improve = 0
            print(f"New best model saved with Val RMSE: {val_rmse:.4f}")
        else:
            no_improve += 5
            print(f"No improvement for {no_improve} epochs. Best Val RMSE: {best_val_rmse:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping after {epoch} epochs!")
            break

print(f"\nTraining completed. Best validation RMSE: {best_val_rmse:.4f}")
print(f"Best model saved to {model_save_path}")

# Save the mapping dictionaries for test inference

with open('models/id_mappings.pkl', 'wb') as f:
    pickle.dump({
        'author_id_map': author_id_map,
        'paper_id_map': paper_id_map
    }, f)

print("ID mappings saved for test inference.")