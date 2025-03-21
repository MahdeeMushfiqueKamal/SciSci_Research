import pandas as pd
import numpy as np

author_df = pd.read_csv("data_raw/Philosophy_2016_Author_Profiles.csv")
papers_df = pd.read_csv("data_raw/Philosophy_2016_Papers.csv")
edges_df = pd.read_csv("data_raw/Philosophy_2016_Edges.csv")


############### Experiment 3: Handling Empty Columns ###############
author_df["Career_Age"] = 2016 - author_df["First_Publication_Year"]
columns_to_drop = [
    "Avg_WSB_sigma",
    "Avg_WSB_mu",
    "Avg_WSB_Cinf",
    "Avg_SB_B",
    "Avg_SB_T",
    "Avg_Reference_Count",
    "Avg_Log10_C5",
    "First_Publication_Year",
    "Total_NSF_Count"
]
author_df.drop(columns=columns_to_drop, inplace=True)

author_df = author_df[author_df["Career_Age"] < 70]
author_ids = set(author_df["AuthorID"])
edges_df = edges_df[edges_df["AuthorID"].isin(author_ids)]
paper_ids = set(edges_df["PaperID"])
papers_df = papers_df[papers_df["PaperID"].isin(paper_ids)]
####################################################################

# Write the author dataset to a CSV file
author_df.to_csv("data/Philosophy_2016_Author_Profiles.csv", index=False)

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
papers_train.to_csv("data/PhilosophyPapers_2016_train.csv", index=False)
papers_validation.to_csv("data/PhilosophyPapers_2016_validation.csv", index=False)
papers_test.to_csv("data/PhilosophyPapers_2016_test.csv", index=False)

# Get paper IDs for each set
train_paper_ids = set(papers_train["PaperID"])
validation_paper_ids = set(papers_validation["PaperID"])
test_paper_ids = set(papers_test["PaperID"])

# Split edges dataset based on paper IDs
edges_train = edges_df[edges_df["PaperID"].isin(train_paper_ids)]
edges_validation = edges_df[edges_df["PaperID"].isin(validation_paper_ids)]
edges_test = edges_df[edges_df["PaperID"].isin(test_paper_ids)]

# Save edges datasets
edges_train.to_csv("data/Philosophy_2016_edges_train.csv", index=False)
edges_validation.to_csv("data/Philosophy_2016_edges_validation.csv", index=False)
edges_test.to_csv("data/Philosophy_2016_edges_test.csv", index=False)

# Print statistics
print(f"Papers total: {len(papers_df)}")
print(
    f"Papers train: {len(papers_train)} ({len(papers_train)/len(papers_df)*100:.1f}%)"
)
print(
    f"Papers validation: {len(papers_validation)} ({len(papers_validation)/len(papers_df)*100:.1f}%)"
)
print(f"Papers test: {len(papers_test)} ({len(papers_test)/len(papers_df)*100:.1f}%)")
print(f"Edges total: {len(edges_df)}")
print(f"Edges train: {len(edges_train)} ({len(edges_train)/len(edges_df)*100:.1f}%)")
print(
    f"Edges validation: {len(edges_validation)} ({len(edges_validation)/len(edges_df)*100:.1f}%)"
)
print(f"Edges test: {len(edges_test)} ({len(edges_test)/len(edges_df)*100:.1f}%)")
