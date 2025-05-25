import os
import pandas as pd
import shutil

# Paths
train_features_csv = ""  # Path to train_features.csv
train_labels_csv = ""  # Path to train_labels.csv
image_folder = ""  # Folder containing images
output_folder = ""  # Folder where organized images will be saved

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the CSV files
features_df = pd.read_csv(train_features_csv)
labels_df = pd.read_csv(train_labels_csv)

# Ensure the 'id' column matches in both dataframes
assert set(features_df['id']) == set(labels_df['id']), "ID mismatch between features and labels CSVs."

# Create directories for each label category
label_columns = labels_df.columns[1:]  # Exclude 'id' column
for label in label_columns:
    label_folder = os.path.join(output_folder, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

# Organize images into folders
for _, row in labels_df.iterrows():
    image_id = row['id']
    image_path = os.path.join(image_folder, f"{image_id}.jpg")  # Image file path

    # Ensure the image exists
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Skipping.")
        continue

    # Find the corresponding label (1 in the row)
    for label in label_columns:
        if row[label] == 1:  # If this label applies
            destination_folder = os.path.join(output_folder, label)
            shutil.copy(image_path, destination_folder)  # Copy image
            print(f"Copied {image_id}.jpg to {destination_folder}")
            break  # Move to the next image once labeled

print("Image organization complete!")
