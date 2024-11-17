import os
import cv2
import pandas as pd

# Path to dataset folders
dataset_folder = 'converted/'  # Adjust the path based on your structure
output_folder = 'generated dataset/'   # Adjust the path based on your structure

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Path to specific subfolders within the dataset
train_folder = os.path.join(dataset_folder, 'train/')
valid_folder = os.path.join(dataset_folder, 'valid/')
test_folder = os.path.join(dataset_folder, 'test/')

# Path to save the CSV files
train_csv = os.path.join(output_folder, 'train.csv')
valid_csv = os.path.join(output_folder, 'valid.csv')
test_csv = os.path.join(output_folder, 'test.csv')

# Function to extract SIFT features
def extract_sift_features(image_path, max_features=100):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to load image {image_path}.")
        return None

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect and compute SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        print(f"Warning: No descriptors found for image {image_path}.")
        return None

    # Flatten the descriptors (or take an average if you want fixed-length features)
    if descriptors.shape[0] > max_features:
        descriptors = descriptors[:max_features]  # Use only the top `max_features`
    flattened = descriptors.flatten()

    # Pad with zeros to ensure all feature vectors are the same length
    if len(flattened) < max_features * descriptors.shape[1]:
        flattened = list(flattened) + [0] * (max_features * descriptors.shape[1] - len(flattened))

    return flattened

# Function to extract features for images in a given folder
def extract_features_from_folder(folder, csv_filename, max_features=100):
    # List to store image data (features + label)
    data = []
    
    # Iterate over the images in the folder
    for image_name in os.listdir(folder):
        # Only process image files
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder, image_name)
            
            # Extract SIFT features for the current image
            features = extract_sift_features(image_path, max_features=max_features)
            if features is not None:
                # Extract the label from the image filename (after the last underscore)
                label = image_name.split('_')[-1].split('.')[0]
                
                # Add label and features to the data list
                row = {'Label': label, 'Image': image_name}
                row.update({f"Feature_{i}": feature for i, feature in enumerate(features)})
                data.append(row)
    
    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"CSV file saved: {csv_filename}")

print("Starting the process...")
# Extract features and save to CSV files for each folder
extract_features_from_folder(train_folder, train_csv)

extract_features_from_folder(valid_folder, valid_csv)
extract_features_from_folder(test_folder, test_csv)
