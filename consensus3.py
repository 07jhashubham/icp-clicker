import os
import json
import cv2
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Load the JSON file
def load_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Load image from the given path
def load_image(image_folder, file_name):
    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {file_name} not found in {image_folder}")
    return image

# Helper function to calculate Mahalanobis distance using robust covariance estimation
def calculate_mahalanobis(bboxes):
    bboxes_array = np.array(bboxes)
    
    # Using MinCovDet for robust covariance estimation
    mcd = MinCovDet().fit(bboxes_array)
    distances = mcd.mahalanobis(bboxes_array)
    
    return distances

# Remove outliers using Mahalanobis distance
def remove_mahalanobis_outliers(bboxes, threshold=3):
    # Remove bounding boxes with NaN or infinite values
    bboxes = [box for box in bboxes if not np.isnan(box).any() and not np.isinf(box).any()]
    
    if len(bboxes) == 0:
        return []
    
    distances = calculate_mahalanobis(bboxes)
    
    return [box for i, box in enumerate(bboxes) if distances[i] < threshold]

# IoU calculation
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# IoU Layer: Assign IoU score to each box
def calculate_iou_layer(bboxes):
    n = len(bboxes)
    iou_scores = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(bboxes[i], bboxes[j])
            iou_scores[i] += iou
            iou_scores[j] += iou
    return iou_scores

# Feature Extraction Layer (SIFT + HOG)
def extract_sift_hog_features(image, bboxes):
    sift = cv2.SIFT_create()
    hog = cv2.HOGDescriptor()

    sift_hog_values = []
    for box in bboxes:
        x, y, w, h = map(int, box)
        cropped_img = image[y:y+h, x:x+w]

        # Ensure the image is converted to grayscale for SIFT and HOG
        if len(cropped_img.shape) == 3:
            cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        else:
            cropped_img_gray = cropped_img

        # SIFT Features
        kp, des = sift.detectAndCompute(cropped_img_gray, None)
        sift_value = np.sum(des) if des is not None else 0

        # Resize the image for HOG to fit expected dimensions
        resized_img = cv2.resize(cropped_img_gray, (64, 128))  # Standard HOG size
        hog_value = hog.compute(resized_img).flatten().sum()

        # Weighted average of SIFT and HOG
        fe_value = 0.5 * sift_value + 0.5 * hog_value
        sift_hog_values.append(fe_value)

    return sift_hog_values

# User Rating Layer (fetch from DB and normalize)
def normalize_user_ratings(user_ratings, max_rating=5.0):
    scaler = MinMaxScaler((0, 1))
    normalized_ratings = scaler.fit_transform(np.array(user_ratings).reshape(-1, 1)).flatten()
    return normalized_ratings

# Bayesian Update Layer
def calculate_bayesian_update(user_ratings, feature_values):
    bayesian_values = []
    for ur, fe in zip(user_ratings, feature_values):
        # Hypothesis is user rating (UR), likelihood is feature value (FE)
        # Simple Bayes update: posterior = prior * likelihood
        posterior = ur * fe
        bayesian_values.append(posterior)
    return bayesian_values

# Platt Layer (apply normalization using MinMaxScaler)
def apply_platt_scaling(iou_values, bayesian_values):
    # Take the product of IoU and Bayesian values
    combined_values = np.array([iou * bayesian for iou, bayesian in zip(iou_values, bayesian_values)])
    
    # Check if there are valid values
    if len(combined_values) == 0 or np.isnan(combined_values).any():
        print("Warning: No valid values for Platt scaling. Returning default values.")
        return np.zeros(len(iou_values))  # Return zeros if no valid values
    
    # Reshape for MinMaxScaler
    combined_values = combined_values.reshape(-1, 1)
    
    # Apply MinMaxScaler to scale the values between 0 and 1
    scaler = MinMaxScaler()
    platt_values = scaler.fit_transform(combined_values).flatten()

    return platt_values

# Final bounding box calculation using Platt-scaled values
def calculate_final_bbox(bboxes, platt_values):
    total_weight = np.sum(platt_values)
    
    if total_weight == 0:
        print("Warning: total_weight is zero. Returning default bounding box.")
        return [0, 0, 0, 0]  # Return a default bounding box
    
    x_avg = np.sum([platt_values[i] * bboxes[i][0] for i in range(len(bboxes))]) / total_weight
    y_avg = np.sum([platt_values[i] * bboxes[i][1] for i in range(len(bboxes))]) / total_weight
    w_avg = np.sum([platt_values[i] * bboxes[i][2] for i in range(len(bboxes))]) / total_weight
    h_avg = np.sum([platt_values[i] * bboxes[i][3] for i in range(len(bboxes))]) / total_weight
    
    return [x_avg, y_avg, w_avg, h_avg]

# Main pipeline to process bounding boxes and calculate final consensus
def process_bboxes_pipeline(image_folder, json_file, user_ratings):
    data = load_json_file(json_file)
    for annotation in data['annotations']:
        image_info = next(img for img in data['images'] if img['id'] == annotation['image_id'])
        image = load_image(image_folder, image_info['file_name'])
        
        # Extract bounding boxes from the annotation
        bboxes = [annotation[f'bbox{i}'] for i in range(1, 5)]
        
        # Step 1: Mahalanobis Outliers
        valid_bboxes = remove_mahalanobis_outliers(bboxes)
        
        # If no valid bounding boxes, skip
        if len(valid_bboxes) == 0:
            print(f"No valid bounding boxes for image {image_info['file_name']}")
            continue
        
        # Step 2: IoU Layer
        iou_values = calculate_iou_layer(valid_bboxes)

        # Step 3: Feature Extraction Layer (SIFT + HOG)
        fe_values = extract_sift_hog_features(image, valid_bboxes)

        # Step 4: User Rating Layer
        ur_values = normalize_user_ratings(user_ratings)

        # Step 5: Bayesian Layer
        bayesian_values = calculate_bayesian_update(ur_values, fe_values)

        # Step 6: Platt Layer
        platt_values = apply_platt_scaling(iou_values, bayesian_values)

        # Step 7: Final Bounding Box
        final_bbox = calculate_final_bbox(valid_bboxes, platt_values)

        print(f"Final bounding box for image {image_info['file_name']}:", final_bbox)

# Example usage
image_folder = 'test'
json_file = 'full_anotation.json'
user_ratings = [4.5, 3.8, 4.0, 2.9]  # Example ratings for the users
process_bboxes_pipeline(image_folder, json_file, user_ratings)
