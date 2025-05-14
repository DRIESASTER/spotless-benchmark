import zarr
from pathlib import Path

zarr_path = Path("spatial_objects/vis.zarr")
z = zarr.open_group(zarr_path, mode='r')

def print_structure(group, indent=""):
    for key in group:
        item = group[key]
        if isinstance(item, zarr.hierarchy.Group):
            print(f"{indent}[Group] {key}")
            print_structure(item, indent + "  ")
        elif isinstance(item, zarr.core.Array):
            print(f"{indent}[Dataset] {key} - Shape: {item.shape}")
        else:
            print(f"{indent}[Unknown] {key}")

print_structure(z)