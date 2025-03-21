import pandas as pd
import shap
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

edges = pd.read_csv("Philosophy_2016_Edges.csv")
papers = pd.read_csv("Philosophy_2016_Papers.csv")

# Sample 10000 papers randomly
sampled_papers = papers.sample(n=10000, random_state=42) 
filtered_edges = edges[edges["PaperID"].isin(sampled_papers["PaperID"])]

print(f"Reduced number of papers: {sampled_papers.shape[0]}")
print(f"Reduced number of authors: {filtered_edges['AuthorID'].nunique()}")

# Create author-paper matrix (Sparse Encoding)
author_paper_matrix = filtered_edges.pivot_table(index="PaperID", columns="AuthorID", aggfunc='size', fill_value=0)

print(f"Author-paper matrix shape: {author_paper_matrix.shape}")


memory_usage = author_paper_matrix.memory_usage(deep=True).sum()
print(f"Memory usage of author-paper matrix: {memory_usage / (1024 ** 2):.2f} MB")

dataset = author_paper_matrix.merge(sampled_papers[['PaperID', 'C5']], on="PaperID").set_index("PaperID")

X = dataset.drop(columns=["C5"])
y = dataset["C5"]

model = xgb.XGBRegressor()
model.fit(X, y)

# Compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

print("\nSHAP values computed")
print(f"SHAP values shape: {shap_values.values.shape}")

shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
print("\nSHAP summary plot saved as 'shap_summary.png'")
