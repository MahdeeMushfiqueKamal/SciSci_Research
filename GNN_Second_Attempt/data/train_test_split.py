import pandas as pd
import numpy as np

papers_df = pd.read_csv('Philosophy_2016_Papers.csv')
edges_df = pd.read_csv('Philosophy_2016_Edges.csv')

np.random.seed(42)
random_values = np.random.rand(len(papers_df))

# Create masks for train (60%), validation (30%), and test (10%)
train_mask = random_values < 0.6
validation_mask = (random_values >= 0.6) & (random_values < 0.9)
test_mask = random_values >= 0.9

# Split papers dataset
papers_train = papers_df[train_mask]
papers_validation = papers_df[validation_mask]
papers_test = papers_df[test_mask]

# Save papers datasets
papers_train.to_csv('PhilosophyPapers_2016_train.csv', index=False)
papers_validation.to_csv('PhilosophyPapers_2016_validation.csv', index=False)
papers_test.to_csv('PhilosophyPapers_2016_test.csv', index=False)

# Get paper IDs for each set
train_paper_ids = set(papers_train['PaperID'])
validation_paper_ids = set(papers_validation['PaperID'])
test_paper_ids = set(papers_test['PaperID'])

# Split edges dataset based on paper IDs
edges_train = edges_df[edges_df['PaperID'].isin(train_paper_ids)]
edges_validation = edges_df[edges_df['PaperID'].isin(validation_paper_ids)]
edges_test = edges_df[edges_df['PaperID'].isin(test_paper_ids)]

# Save edges datasets
edges_train.to_csv('Philosophy_2016_edges_train.csv', index=False)
edges_validation.to_csv('Philosophy_2016_edges_validation.csv', index=False)
edges_test.to_csv('Philosophy_2016_edges_test.csv', index=False)

# Print statistics
print(f"Papers total: {len(papers_df)}")
print(f"Papers train: {len(papers_train)} ({len(papers_train)/len(papers_df)*100:.1f}%)")
print(f"Papers validation: {len(papers_validation)} ({len(papers_validation)/len(papers_df)*100:.1f}%)")
print(f"Papers test: {len(papers_test)} ({len(papers_test)/len(papers_df)*100:.1f}%)")
print(f"Edges total: {len(edges_df)}")
print(f"Edges train: {len(edges_train)} ({len(edges_train)/len(edges_df)*100:.1f}%)")
print(f"Edges validation: {len(edges_validation)} ({len(edges_validation)/len(edges_df)*100:.1f}%)")
print(f"Edges test: {len(edges_test)} ({len(edges_test)/len(edges_df)*100:.1f}%)")