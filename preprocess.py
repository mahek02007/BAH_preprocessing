import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# ---------------------------
# Helper functions
# ---------------------------
def normalize(band):
    """Scale band values to 0–1 for visualization."""
    return (band - band.min()) / (band.max() - band.min())

def create_tiles(image, tile_size=256):
    """Split image into non-overlapping tiles."""
    tiles = []
    for i in range(0, image.shape[0], tile_size):
        for j in range(0, image.shape[1], tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            if tile.shape == (tile_size, tile_size, 3) or tile.shape == (tile_size, tile_size):
                tiles.append(tile)
    return tiles
def save_tiles(tiles, folder, prefix="tile"):
    """Save tiles as PNG images (handles 2D and 3D)."""
    os.makedirs(folder, exist_ok=True)
    for idx, tile in enumerate(tiles):
        if tile.ndim == 2:  # grayscale (NDVI)
            out = (tile * 255).astype("uint8")
            cv2.imwrite(f"{folder}/{prefix}_{idx}.png", out)
        elif tile.ndim == 3:  # RGB or FalseColor
            out = (tile * 255).astype("uint8")
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)  # <-- key fix
            cv2.imwrite(f"{folder}/{prefix}_{idx}.png", out)




# ---------------------------
# Load bands
# --------------------------

blue = rasterio.open("LC09_L2SP_147040_20260617_20260622_02_T1_SR_B2.TIF").read(1)
green = rasterio.open("LC09_L2SP_147040_20260617_20260622_02_T1_SR_B3.TIF").read(1)
red = rasterio.open("LC09_L2SP_147040_20260617_20260622_02_T1_SR_B4.TIF").read(1)
nir = rasterio.open("LC09_L2SP_147040_20260617_20260622_02_T1_SR_B5.TIF").read(1)
# Normalize function (scales values to 0–1)
def normalize(band):
    return (band - band.min()) / (band.max() - band.min())

# Apply normalization
red_norm = normalize(red)
green_norm = normalize(green)
blue_norm = normalize(blue)

# Stack into RGB composite
rgb = np.dstack((red_norm, green_norm, blue_norm))

# Plot
plt.imshow(rgb)
plt.title("True Color Composite (Normalized)")
plt.show()
def stretch(band, lower=2, upper=98):
    # Percentile stretch
    min_val, max_val = np.percentile(band, (lower, upper))
    return np.clip((band - min_val) / (max_val - min_val), 0, 1)
false_color = np.dstack((normalize(nir), normalize(red), normalize(green)))
plt.imshow(false_color)
plt.title("False Color Composite")
plt.show()


rgb_stretched = np.dstack((stretch(red), stretch(green), stretch(blue)))
plt.imshow(rgb_stretched)
plt.title("True Color Composite (Contrast Stretched)")
plt.show()
from rasterio import open as rio_open
from rasterio.transform import Affine
from rasterio.crs import CRS

# Example: save normalized RGB composite
with rio_open(
    "true_color.tif",
    "w",
    driver="GTiff",
    height=rgb.shape[0],
    width=rgb.shape[1],
    count=3,
    dtype="float32",
    crs=CRS.from_epsg(4326),  # WGS84
    transform=Affine.identity()
) as dst:
    dst.write((red_norm).astype("float32"), 1)
    dst.write((green_norm).astype("float32"), 2)
    dst.write((blue_norm).astype("float32"), 3)
ndvi = (nir - red) / (nir + red)
plt.imshow(ndvi, cmap='RdYlGn')
plt.title("NDVI (Vegetation Index)")
plt.colorbar()
plt.show()
# ---------------------------
# Tiling & Saving
# ---------------------------
rgb_tiles = create_tiles(rgb, 256)
false_tiles = create_tiles(false_color, 256)
ndvi_tiles = create_tiles(ndvi, 256)
save_tiles(rgb_tiles, "tiles/RGB", "rgb")
save_tiles(false_tiles, "tiles/FalseColor", "false")
save_tiles(ndvi_tiles, "tiles/NDVI", "ndvi")



import random, shutil

def split_dataset(src_folder, train_ratio=0.7, val_ratio=0.2):
    files = os.listdir(src_folder)
    random.shuffle(files)
    train_end = int(train_ratio * len(files))
    val_end = train_end + int(val_ratio * len(files))

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for subset, subset_files in zip(
        ["train", "val", "test"], [train_files, val_files, test_files]
    ):
        subset_folder = os.path.join(src_folder, subset)
        os.makedirs(subset_folder, exist_ok=True)
        for f in subset_files:
            shutil.move(os.path.join(src_folder, f), os.path.join(subset_folder, f))
with rio_open(
    "ndvi.tif",
    "w",
    driver="GTiff",
    height=ndvi.shape[0],
    width=ndvi.shape[1],
    count=1,
    dtype="float32",
    crs=CRS.from_epsg(4326),
    transform=Affine.identity()
) as dst:
    dst.write(ndvi.astype("float32"), 1)
print(" Bands loaded")
print(" Composites generated")
print(" NDVI calculated")
print(" Tiles created and saved")


import os

def verify_structure(base_folder="tiles"):
    for root, dirs, files in os.walk(base_folder):
        level = root.replace(base_folder, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        if files:
            print(f"{subindent}--> {len(files)} files")

verify_structure("tiles")
print("False color shape:", false_color.shape)
