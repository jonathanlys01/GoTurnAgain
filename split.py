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
num_videos = len(video_dirs)

# Split the videos into training and validation sets (70% train, 30% val)
num_train = int(0.7 * num_videos)

# Randomly shuffle the video directories
random.shuffle(video_dirs)

# Split the video directories into training and validation sets
train_dirs = video_dirs[:num_train]
val_dirs = video_dirs[num_train:]

# Save the results to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video', 'set'])
    for video in train_dirs:
        writer.writerow([video, 'train'])
    for video in val_dirs:
        writer.writerow([video, 'val'])

print(f'Saved the results to {output_csv}')

