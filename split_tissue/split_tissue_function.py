import pandas as pd
import numpy as np
import scanpy as sc 
import matplotlib.pyplot as plt
import os
import argparse

def split_tissue(input_pipeline_a, spatial_coords_a, n, figure=True, output_dir='output_files'):
    ''' 
        Splits tissue samples into a grid of n*n parts based on the spatial coordinates of the spots. 
        This function generates n*n AnnData objects, an updated AnnData object with assigned parts, a CSV file containing the spots in each part, and optionally a figure showing the distribution of spots across the parts.

        Args:
            input_pipeline (str): File path to an AnnData object stored in .h5ad format or .h5 format, containing data for the spots.
            spatial_coords (str): File path to a CSV file with 6 columns: 'spot_id', 'tissue', 'x0', 'y0', 'x', 'y', which define the spatial coordinates of the spots.
            n (int): The number of divisions per dimension, with the tissue being split into an n*n grid of parts.
            figure (bool, optional): If `True`, a figure will be displayed showing the distribution of spots across the parts (default is `True`).
            output_dir (str, optional): Directory where the output files will be saved (default is 'output_files').

        Returns:
            None: The function saves the updated AnnData objects as .h5ad files, generates a CSV file with the spots' part assignments, 
            and shows a figure if `figure=True`.

        Notes:
            - The parts are numbered from 0 to n*n-1. Part 0 contains the spots with the smallest x and y coordinates, part 1 contains the spots with the smallest x and the second smallest y coordinates, and so on.
            - The function divides the spots evenly across the n*n grid based on their spatial coordinates.
            - It checks whether all spots from the `input_pipeline` file are present in the `spatial_coords` dataset.
            
        Requirements:
            - The function requires the following libraries to be installed: pandas, numpy, scanpy, matplotlib.
    '''

    # Create the output directory if it doesn't exist, and overwrite if it exists
    if os.path.exists(output_dir):
        # Remove existing files in the output directory before saving new ones
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_dir)

    # Load data
    spatial_coords = pd.read_csv(spatial_coords_a, delimiter=',', names=['spot_id', 'tissue', 'x0', 'y0', 'x', 'y'])
    if input_pipeline_a.endswith('.h5ad'):
        input_pipeline = sc.read_h5ad(input_pipeline_a)
    elif input_pipeline_a.endswith('.h5'):
        input_pipeline = sc.read_10x_h5(input_pipeline_a)
    else:
        print('Invalid file format for input_pipeline')
        return None
    input_pipeline.obs['part'] = np.nan

    # Check whether all spots in the input_pipeline file are present in the spatial_coords file
    if input_pipeline_a.endswith('.h5ad'):
        if not input_pipeline.obs.index.str[:-2].isin(spatial_coords['spot_id']).all():
            print('Not all spots are present in the spatial_coords file')
            return None
    else:
        if not input_pipeline.obs.index.isin(spatial_coords['spot_id']).all():
            print('Not all spots are present in the spatial_coords file')
            return None
    
    # Filter the spatial_coords file to only include the spots present in the input_pipeline file
    if input_pipeline_a.endswith('.h5ad'):
        common_spots = input_pipeline.obs.index.str[:-2].intersection(spatial_coords.iloc[:,0])
    else:
        common_spots = input_pipeline.obs.index.intersection(spatial_coords.iloc[:,0])
    spatial_coords_filtered = spatial_coords[spatial_coords['spot_id'].isin(common_spots)]

    # Initialize part column with -1
    count = 0

    # Sort the spots by x and y coordinates
    sorted_spots_x = spatial_coords_filtered.sort_values(['x0', 'y0'])

    # Split the x and y coordinates into n equal parts
    x_bins = np.array_split(sorted_spots_x, n, axis=0)

    # Iterate through the bins to assign parts
    for i in range(n):
        x_subset = x_bins[i]

        # Sort the subset by y and x for the next step
        sorted_subset_y = x_subset.sort_values(['y', 'x'])
        y_bins = np.array_split(sorted_subset_y, n, axis=0)

        for j in range(n):            
            # Get the spot ids that belong to this part
            q = y_bins[j]['spot_id']

            # Update the 'part' values in both input_pipeline and spatial_coords_filtered
            if input_pipeline_a.endswith('.h5ad'):
                input_pipeline.obs.loc[input_pipeline.obs.index.str[:-2].isin(q), 'part'] = count
            else:
                input_pipeline.obs.loc[input_pipeline.obs.index.isin(q), 'part'] = count
            spatial_coords_filtered.loc[spatial_coords_filtered['spot_id'].isin(q), 'part'] = count
            count += 1

    # Control whether all spots are assigned to a part
    if sum(pd.isna(input_pipeline.obs['part'])) > 0:
        print('Not all spots are assigned to a part')
    else:
        print('All spots are assigned to a part')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the updated AnnData object (AnnData object, containing the parts) to a .h5ad file
    if input_pipeline_a.endswith('.h5ad'):
        input_pipeline.__dict__['_raw'].__dict__['_var'] = input_pipeline.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
    
    output_file = "updated_file.h5ad"
    file_name2 = os.path.join(output_dir, "updated_annData_obj.h5ad")
    input_pipeline.write(filename=file_name2)
    print(f"Saved updated AnnData object to {file_name2}")

    # Create a CSV file with the part each spot belongs to
    df = pd.DataFrame(input_pipeline.obs['part'])
    df.index = input_pipeline.obs['part'].index
    csv_file = os.path.join(output_dir, "spots_and_parts.csv")
    df.to_csv(csv_file)
    print(f"Saved the index of the spots in each part to {csv_file}")

    # Create a new AnnData object for each part
    for part_value in sorted(input_pipeline.obs['part'].unique()):
        part_data = input_pipeline[input_pipeline.obs['part'] == part_value]
        part_data.obs = part_data.obs.drop(columns=["part"])
    
        # Save each part as a separate .h5ad file
        part_file_name = os.path.join(output_dir, f"annData_part_{int(part_value)}.h5ad")
        part_data.write(filename=part_file_name)
        print(f"Saved AnnData object for part {part_value} to {part_file_name}")


    if figure == True:
        # Calculate the count for each part
        part_counts = input_pipeline.obs['part'].value_counts()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))

        # Create the bar plot
        bars = ax.bar(sorted(part_counts.index.astype(str)), part_counts.values, color='salmon')

        # Set labels and title
        ax.set_xlabel('Part')
        ax.set_ylabel('Number of spots')
        ax.set_title('Distribution of spots across parts')
        ax.set_xticks(sorted(part_counts.index.astype(int)))
        ax.set_xticklabels(sorted(part_counts.index.astype(int)))

        # Display the height of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, str(int(yval)), ha='center', va='bottom')

        # Save the figure
        plt.savefig("distribution_plot.png", dpi=300, bbox_inches='tight')

    return None

def main():
    parser = argparse.ArgumentParser(description="Split tissue into parts.")
    parser.add_argument("input_pipeline", help="Path to AnnData .h5ad file")
    parser.add_argument("spatial_coords", help="Path to spatial coordinates CSV file")
    parser.add_argument("n", type=int, help="Number of divisions per dimension")
    parser.add_argument("--figure", type=bool, default=True, help="Whether to show the figure")
    parser.add_argument("--output_dir", default="output_files", help="Output directory")
    args = parser.parse_args()

    split_tissue(args.input_pipeline, args.spatial_coords, args.n, args.figure, args.output_dir)

if __name__ == "__main__":
    main()
