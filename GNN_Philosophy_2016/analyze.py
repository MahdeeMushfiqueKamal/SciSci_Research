#!/usr/bin/env python3
"""
analyze_results.py - Analyze citation prediction results

This script analyzes a CSV file containing paper citation predictions,
comparing actual citation counts with predicted counts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_data(file_path):
    """Load and validate the citation prediction data."""
    print(f"Loading data from {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check for required columns
    required_cols = ['PaperID', 'Actual_C5', 'Predicted_C5']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Basic cleaning - remove any NaN values
    original_count = len(df)
    df = df.dropna()
    
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} rows with missing values.")
    
    return df

def calculate_statistics(df):
    """Calculate basic statistics about the predictions."""
    stats_dict = {
        'Total Papers': len(df),
        'Zero Citation Papers': (df['Actual_C5'] == 0).sum(),
        'Zero Citation Papers (%)': (df['Actual_C5'] == 0).mean() * 100,
        'Mean Actual Citations': df['Actual_C5'].mean(),
        'Median Actual Citations': df['Actual_C5'].median(),
        'Max Actual Citations': df['Actual_C5'].max(),
        'Mean Predicted Citations': df['Predicted_C5'].mean(),
        'Median Predicted Citations': df['Predicted_C5'].median(),
        'Max Predicted Citations': df['Predicted_C5'].max(),
        'Correlation': df['Actual_C5'].corr(df['Predicted_C5']),
        'Mean Absolute Error': (df['Actual_C5'] - df['Predicted_C5']).abs().mean(),
        'Median Absolute Error': (df['Actual_C5'] - df['Predicted_C5']).abs().median(),
        'Root Mean Square Error': np.sqrt(((df['Actual_C5'] - df['Predicted_C5']) ** 2).mean()),
        'Accuracy (within ±10)': (abs(df['Actual_C5'] - df['Predicted_C5']) <= 10).mean() * 100
    }
    
    return stats_dict

def print_statistics(stats_dict):
    """Print the calculated statistics in a formatted way."""
    print("\n===== CITATION PREDICTION STATISTICS =====")
    
    # Dataset overview
    print("\nDATASET OVERVIEW:")
    print(f"Total Papers: {stats_dict['Total Papers']:,}")
    print(f"Papers with Zero Citations: {stats_dict['Zero Citation Papers']:,} ({stats_dict['Zero Citation Papers (%)']:.2f}%)")
    
    # Citation metrics
    print("\nCITATION METRICS:")
    print(f"Average Actual Citations: {stats_dict['Mean Actual Citations']:.2f}")
    print(f"Median Actual Citations: {stats_dict['Median Actual Citations']:.1f}")
    print(f"Maximum Actual Citations: {stats_dict['Max Actual Citations']:.0f}")
    print(f"Average Predicted Citations: {stats_dict['Mean Predicted Citations']:.2f}")
    print(f"Median Predicted Citations: {stats_dict['Median Predicted Citations']:.1f}")
    print(f"Maximum Predicted Citations: {stats_dict['Max Predicted Citations']:.0f}")
    print(f"Correlation: {stats_dict['Correlation']:.3f}")
    
    # Prediction performance
    print("\nPREDICTION PERFORMANCE:")
    print(f"Mean Absolute Error: {stats_dict['Mean Absolute Error']:.2f}")
    print(f"Median Absolute Error: {stats_dict['Median Absolute Error']:.2f}")
    print(f"Root Mean Square Error: {stats_dict['Root Mean Square Error']:.2f}")
    print(f"Accuracy (within ±10): {stats_dict['Accuracy (within ±10)']:.2f}%")

def analyze_extreme_cases(df):
    """Find and analyze papers with extreme prediction errors."""
    # Calculate error
    df['Error'] = df['Predicted_C5'] - df['Actual_C5']
    
    # Find largest overpredictions and underpredictions
    largest_over = df.nlargest(5, 'Error')
    largest_under = df.nsmallest(5, 'Error')
    
    print("\nTOP 5 OVERPREDICTED PAPERS:")
    for _, row in largest_over.iterrows():
        print(f"Paper ID: {row['PaperID']}, Actual: {row['Actual_C5']:.1f}, "
              f"Predicted: {row['Predicted_C5']:.1f}, Error: {row['Error']:.1f}")
    
    print("\nTOP 5 UNDERPREDICTED PAPERS:")
    for _, row in largest_under.iterrows():
        print(f"Paper ID: {row['PaperID']}, Actual: {row['Actual_C5']:.1f}, "
              f"Predicted: {row['Predicted_C5']:.1f}, Error: {row['Error']:.1f}")
    
    return df

def create_binned_data(df, max_bin=50):
    """Create binned data for analysis by actual citation count."""
    bins = {}
    
    # Group data by actual citation count
    for _, row in df.iterrows():
        actual = row['Actual_C5']
        bin_key = min(int(actual), max_bin)
        
        if bin_key not in bins:
            bins[bin_key] = {
                'count': 0,
                'total_predicted': 0,
                'min_predicted': float('inf'),
                'max_predicted': float('-inf')
            }
        
        bins[bin_key]['count'] += 1
        bins[bin_key]['total_predicted'] += row['Predicted_C5']
        bins[bin_key]['min_predicted'] = min(bins[bin_key]['min_predicted'], row['Predicted_C5'])
        bins[bin_key]['max_predicted'] = max(bins[bin_key]['max_predicted'], row['Predicted_C5'])
    
    # Convert to DataFrame
    bin_data = []
    for i in range(max_bin + 1):
        if i in bins:
            avg_predicted = bins[i]['total_predicted'] / bins[i]['count']
            bin_data.append({
                'actual_citations': i if i < max_bin else f"{max_bin}+",
                'avg_predicted': avg_predicted,
                'count': bins[i]['count'],
                'min_predicted': bins[i]['min_predicted'],
                'max_predicted': bins[i]['max_predicted']
            })
        elif i < max_bin:
            bin_data.append({
                'actual_citations': i,
                'avg_predicted': 0,
                'count': 0,
                'min_predicted': 0,
                'max_predicted': 0
            })
    
    bin_df = pd.DataFrame(bin_data)
    return bin_df

def create_visualizations(df, bin_df, output_dir='plots'):
    """Create various visualizations to analyze the prediction data."""
    print(f"\nCreating visualizations in '{output_dir}' directory...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Actual vs Predicted scatter plot (sample for readability)
    plt.figure(figsize=(10, 8))
    
    # Sample the data if there are too many points
    sample_size = 1000 if len(df) > 1000 else len(df)
    sample_df = df.sample(sample_size) if len(df) > sample_size else df
    
    # Create scatter plot
    ax = sns.scatterplot(x='Actual_C5', y='Predicted_C5', data=sample_df, alpha=0.6)
    
    # Add perfect prediction line
    max_val = max(df['Actual_C5'].max(), df['Predicted_C5'].max())
    perfect_line = np.linspace(0, min(50, max_val), 100)
    plt.plot(perfect_line, perfect_line, 'r--', label='Perfect Prediction')
    
    # Limit axes for better visibility
    plt.xlim(-1, 50)
    plt.ylim(-1, 50)
    
    plt.title('Actual vs. Predicted Citations')
    plt.xlabel('Actual Citations')
    plt.ylabel('Predicted Citations')
    plt.legend()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution histogram
    plt.figure(figsize=(12, 8))
    df['Error'] = df['Predicted_C5'] - df['Actual_C5']
    
    # Filter extreme values for better visualization
    filtered_errors = df[(df['Error'] > -50) & (df['Error'] < 50)]['Error']
    
    sns.histplot(filtered_errors, bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f"{output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Average predictions by actual citation count
    plt.figure(figsize=(12, 8))
    
    # Convert string values to numeric for plotting
    bin_df['actual_numeric'] = bin_df['actual_citations'].apply(
        lambda x: int(x.replace('+', '')) if isinstance(x, str) else x
    )
    
    # Sort by actual citations
    bin_df = bin_df.sort_values('actual_numeric')
    
    # Plot average predictions
    plt.plot(bin_df['actual_numeric'], bin_df['avg_predicted'], 'bo-', label='Avg Predicted')
    
    # Add perfect prediction line
    plt.plot(bin_df['actual_numeric'], bin_df['actual_numeric'], 'r--', label='Perfect Prediction')
    
    plt.title('Average Predicted Citations by Actual Citation Count')
    plt.xlabel('Actual Citations')
    plt.ylabel('Average Predicted Citations')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/avg_predictions_by_actual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Citation count distribution
    plt.figure(figsize=(10, 8))
    
    # Create a new dataframe with counts up to a certain value
    max_display = 20
    citation_counts = pd.DataFrame({
        'Citations': range(max_display + 1),
        'Count': [sum(df['Actual_C5'] == i) for i in range(max_display + 1)]
    })
    
    # Plot as bar chart
    sns.barplot(x='Citations', y='Count', data=citation_counts)
    plt.yscale('log')  # Log scale for better visibility
    plt.title('Distribution of Citation Counts')
    plt.xlabel('Citation Count')
    plt.ylabel('Number of Papers (log scale)')
    plt.savefig(f"{output_dir}/citation_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to '{output_dir}' directory.")

def main():
    """Main function to run the analysis."""
    try:
        # Define the input file path
        file_path = 'citation_predictions.csv'
        
        # Load the data
        df = load_data(file_path)
        
        # Calculate statistics
        stats = calculate_statistics(df)
        
        # Print statistics
        print_statistics(stats)
        
        # Analyze extreme cases
        df = analyze_extreme_cases(df)
        
        # Create binned data
        binned_data = create_binned_data(df)
        
        # Create visualizations
        create_visualizations(df, binned_data)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())