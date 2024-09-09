import json
import random

# Function to add noise to the bounding box
def add_noise_to_bbox(bbox, noise_factor=0.1):
    x, y, width, height = bbox
    noise_x = random.uniform(-noise_factor, noise_factor) * width
    noise_y = random.uniform(-noise_factor, noise_factor) * height
    noise_w = random.uniform(-noise_factor, noise_factor) * width
    noise_h = random.uniform(-noise_factor, noise_factor) * height
    
    new_bbox = [
        max(0, x + noise_x),  # Ensure values remain positive
        max(0, y + noise_y),
        max(0, width + noise_w),
        max(0, height + noise_h)
    ]
    
    return new_bbox

# Function to generate an outlier bounding box (drastically different)
def generate_outlier_bbox(bbox):
    x, y, width, height = bbox
    # Generate a drastically different bbox with random offsets
    outlier_x = random.uniform(-2, 2) * width + x  # Larger shift
    outlier_y = random.uniform(-2, 2) * height + y
    outlier_width = max(0, width * random.uniform(0.2, 2.0))  # Large size variation
    outlier_height = max(0, height * random.uniform(0.2, 2.0))
    
    outlier_bbox = [
        max(0, outlier_x),  # Ensure values remain positive
        max(0, outlier_y),
        outlier_width,
        outlier_height
    ]
    
    return outlier_bbox

# Function to calculate the area of the bounding box
def calculate_area(bbox):
    _, _, width, height = bbox
    return width * height

# Load JSON file
with open('_annotations.coco.json', 'r') as f:
    data = json.load(f)

# Iterate through annotations and modify the bbox property
for i, annotation in enumerate(data['annotations']):
    original_bbox = annotation.pop('bbox')  # Remove the original bbox

    # Generate four noisy bounding boxes
    annotation['bbox1'] = add_noise_to_bbox(original_bbox)
    annotation['bbox2'] = add_noise_to_bbox(original_bbox)
    annotation['bbox3'] = add_noise_to_bbox(original_bbox)
    annotation['bbox4'] = add_noise_to_bbox(original_bbox)

    # Add outliers for every 3-4 images and 2 outliers every 10-12 images
    if (i % 3 == 0 or i % 4 == 0):  # For every 3-4 images, add one outlier
        outlier_index = random.choice([1, 2, 3, 4])  # Randomly pick one bbox to replace
        annotation[f'bbox{outlier_index}'] = generate_outlier_bbox(original_bbox)

    if (i % 10 == 0 or i % 12 == 0):  # For every 10-12 images, add two outliers
        outlier_indices = random.sample([1, 2, 3, 4], 2)  # Pick two random boxes
        annotation[f'bbox{outlier_indices[0]}'] = generate_outlier_bbox(original_bbox)
        annotation[f'bbox{outlier_indices[1]}'] = generate_outlier_bbox(original_bbox)

    # Recalculate area for each bounding box
    annotation['area1'] = calculate_area(annotation['bbox1'])
    annotation['area2'] = calculate_area(annotation['bbox2'])
    annotation['area3'] = calculate_area(annotation['bbox3'])
    annotation['area4'] = calculate_area(annotation['bbox4'])

# Save the modified data back to a new JSON file
with open('modified_annotations_with_outliers.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Bounding boxes modified with outliers and saved to modified_annotations_with_outliers.json")
