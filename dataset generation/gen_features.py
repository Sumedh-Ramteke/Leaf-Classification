import os
import csv
import cv2
import pandas as pd
from features1 import extract_features

# Path to your folders
train_folder = 'converted/train/'
valid_folder = 'converted/valid/'
test_folder = 'converted/test/'

# Path to save the CSV files
train_csv = 'train.csv'
valid_csv = 'valid.csv'
test_csv = 'test.csv'

# Function to extract features for images in a given folder
def extract_features_from_folder(folder, csv_filename):
    # List to store image data (features + label)
    data = []
    
    # Iterate over the images in the folder
    for image_name in os.listdir(folder):
        # Only process image files
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder, image_name)
            
            # Extract features for the current image
            features = extract_features(image_path)
            if features is not None:
                # Extract the label from the image filename (after the last underscore)
                label = image_name.split('_')[-1].split('.')[0]
                
                # Add label and features to the data list
                row = features
                row['Label'] = label  # Append the label as the last column
                row['Image'] = image_name  # Add image filename to row
                data.append(row)
    
    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"CSV file saved: {csv_filename}")

# Extract features and save to CSV files for each folder
extract_features_from_folder(train_folder, train_csv)
extract_features_from_folder(valid_folder, valid_csv)
extract_features_from_folder(test_folder, test_csv)
