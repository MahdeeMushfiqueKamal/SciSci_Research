import pandas as pd
import numpy as np

papers_df = pd.read_csv('PhilosophyPapers_2016.csv')
edges_df = pd.read_csv('Philosophy_2016_edges.csv')

np.random.seed(42)
mask = np.random.rand(len(papers_df)) < 0.5

papers_validation = papers_df[mask]
papers_test = papers_df[~mask]

papers_validation.to_csv('PhilosophyPapers_2016_validation.csv', index=False)
papers_test.to_csv('PhilosophyPapers_2016_test.csv', index=False)

validation_paper_ids = set(papers_validation['PaperID'])
test_paper_ids = set(papers_test['PaperID'])

edges_validation = edges_df[edges_df['PaperID'].isin(validation_paper_ids)]
edges_test = edges_df[edges_df['PaperID'].isin(test_paper_ids)]

edges_validation.to_csv('Philosophy_2016_edges_validation.csv', index=False)
edges_test.to_csv('Philosophy_2016_edges_test.csv', index=False)

print(f"Papers total: {len(papers_df)}")
print(f"Papers validation: {len(papers_validation)} ({len(papers_validation)/len(papers_df)*100:.1f}%)")
print(f"Papers test: {len(papers_test)} ({len(papers_test)/len(papers_df)*100:.1f}%)")
print(f"Edges total: {len(edges_df)}")
print(f"Edges validation: {len(edges_validation)} ({len(edges_validation)/len(edges_df)*100:.1f}%)")
print(f"Edges test: {len(edges_test)} ({len(edges_test)/len(edges_df)*100:.1f}%)")