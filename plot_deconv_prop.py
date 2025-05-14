import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


def plot_cell_distribution(filename, output_dir=None):
    """Generates a bar plot of cell type proportions for the given file and optionally saves it."""
    df = pd.read_csv(filename, sep="\t")
    # Calculate proportions for each row and then the mean across rows
    proportions = df.div(df.sum(axis=1), axis=0).mean()

    plt.figure(figsize=(12, 6))
    proportions.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Cell Type")
    plt.ylabel("Proportion")
    plt.title(f"Proportional Distribution of Cell Types: {os.path.basename(filename)}")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(filename)}.png")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def aggregate_and_plot(input_dir, output_file):
    """Aggregates all data from the input directory and generates a combined plot."""
    all_files = glob(os.path.join(input_dir, '*'))
    if not all_files:
        raise ValueError(f"No files found in directory: {input_dir}")

    # Read and concatenate all data
    combined_df = pd.concat([pd.read_csv(f, sep='\t') for f in all_files], ignore_index=True)
    # Calculate proportions for each row and then the mean across all rows
    proportions = combined_df.div(combined_df.sum(axis=1), axis=0).mean()

    plt.figure(figsize=(12, 6))
    proportions.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Cell Type")
    plt.ylabel("Proportion")
    plt.title("Aggregated Proportional Distribution of Cell Types")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    input_directory = 'deconv_proportions'
    plot_output_directory = 'cell_type_plots'
    aggregated_output_path = 'aggregated_cell_distribution.png'

    # Generate individual plots for each file
    all_files = glob(os.path.join(input_directory, '*'))
    for file_path in all_files:
        plot_cell_distribution(file_path, plot_output_directory)

    # Generate aggregated plot
    aggregate_and_plot(input_directory, aggregated_output_path)