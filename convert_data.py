import os
import cv2
import yaml

# Path to your image folder, annotation folder, and the data.yaml file
image_folder = 'roboflow/test/images'
annotation_folder = 'roboflow/test/labels'
output_folder = 'converted/test'
yaml_file = 'data.yaml'

# Load class names from data.yaml
with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

# List of species names (this is read from the data.yaml)
class_labels = data['names']

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to crop the image based on YOLO annotation
def crop_and_save_image(image_path, annotation_path):
    # Read the image
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape
    
    # Read the YOLO annotation file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    # Process each bounding box in the annotation file
    for i, line in enumerate(annotations):
        # Parse the line in YOLO format
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        
        # Convert to top-left and bottom-right corner coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        
        # Ensure that the crop coordinates are within bounds of the image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Get the label for the cropped image based on the class_id
        label = class_labels[class_id]

        # Save the cropped image with the label as the filename
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{label}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved cropped image: {output_path}")

# Process each image in the image folder
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.png')):
        # Construct the full image path and annotation path
        image_path = os.path.join(image_folder, image_name)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_name)[0] + '.txt')
        
        if os.path.exists(annotation_path):
            crop_and_save_image(image_path, annotation_path)
        else:
            print(f"Warning: No annotation file found for {image_name}")