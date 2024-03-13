"""
Split the ALOV300 dataset into training and validation sets. Save the results to a CSV file.
"""

import os
import csv
import random
import numpy as np

# Set the random seed for reproducibility
random.seed(0)

# Set the path to the ALOV300 dataset
from config import cfg
alov_path = os.path.join(cfg.paths.alov, 'imagedata++')

# Set the path to the output CSV file
output_csv = os.path.join('data/alov300_split.csv')

# Get the list of all video directories
video_dirs = [d for d in os.listdir(alov_path) if os.path.isdir(os.path.join(alov_path, d))]
video_paths = []
for video_dir in video_dirs:
    video_paths.extend([os.path.join(video_dir, d) for d in os.listdir(os.path.join(alov_path, video_dir)) if os.path.isdir(os.path.join(alov_path, video_dir, d))])
num_videos = len(video_paths)

# Split the videos into training and validation sets (70% train, 30% val)
num_train = int(0.7 * num_videos)

# Randomly shuffle the video paths
random.shuffle(video_paths)

# Split the video directories into training and validation sets
train_paths = video_paths[:num_train]
val_paths = video_paths[num_train:]

# Save the results to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video', 'split'])
    for video in train_paths:
        writer.writerow([video, 'train'])
    for video in val_paths:
        writer.writerow([video, 'val'])

print(f'Saved the results to {output_csv}')

