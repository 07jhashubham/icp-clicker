import numpy as np
import cv2
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import json
import os
import torch
import torchvision
from torchvision import models, transforms
import torch.nn.functional as F

def remove_outliers(boxes):
    """
    Remove outlier bounding boxes using Mahalanobis distance with regularization.
    """
    data = np.array(boxes)
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # Add a small regularization term to the diagonal of the covariance matrix
    cov += np.eye(cov.shape[0]) * 1e-6

    # Check if the covariance matrix is still singular
    if np.linalg.cond(cov) > 1e12:
        print("Covariance matrix is close to singular. Using pseudo-inverse.")
        inv_covmat = np.linalg.pinv(cov)
    else:
        inv_covmat = np.linalg.inv(cov)

    distances = []
    for i in range(len(data)):
        distance = mahalanobis(data[i], mean, inv_covmat)
        distances.append(distance)

    # Determine threshold (e.g., 95% confidence interval)
    threshold = chi2.ppf(0.95, df=4)
    inliers = [boxes[i] for i in range(len(boxes)) if distances[i] <= threshold]
    return inliers

def compute_iou_scores(boxes):
    """
    Compute IoU scores for each bounding box.
    """
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    num_boxes = len(boxes)
    iou_scores = np.zeros(num_boxes)
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j:
                iou_scores[i] += iou(boxes[i], boxes[j])
    # Normalize IoU scores
    scaler = MinMaxScaler()
    iou_scores = scaler.fit_transform(iou_scores.reshape(-1, 1)).flatten()
    return iou_scores

def extract_features(image, boxes):
    """
    Extract CNN features for each bounding box using a pre-trained ResNet50 model.
    """
    # Load a pre-trained CNN model
    model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    
    # Remove the final classification layer to get features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    
    features_list = []
    height, width = image.shape[:2]
    for box in boxes:
        x, y, w, h = map(int, box)
        x_end = min(x + w, width)
        y_end = min(y + h, height)
        x = max(x, 0)
        y = max(y, 0)
        w = x_end - x
        h = y_end - y
        roi = image[y:y_end, x:x_end]
        
        if roi.size == 0:
            print(f"Empty ROI for box: {box}")
            features_list.append(np.zeros(2048))  # ResNet50 feature size
            continue
        
        # Preprocess ROI
        input_tensor = preprocess(roi)
        input_batch = input_tensor.unsqueeze(0)  # Create batch dimension
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
        
        with torch.no_grad():
            output = model(input_batch)
        # Flatten the output
        features = output.cpu().numpy().flatten()
        features_list.append(features)
    
    # Now, features_list is a list of feature vectors
    # For simplicity, we'll compute the L2 norm of each feature vector as a scalar value
    fe_values = [np.linalg.norm(f) for f in features_list]
    # Normalize features
    scaler = MinMaxScaler()
    fe_values = scaler.fit_transform(np.array(fe_values).reshape(-1, 1)).flatten()
    return fe_values

def bayesian_update(ur_values, fe_values):
    """
    Compute posterior probabilities using Bayesian update.
    """
    # Normalize UR and FE values
    scaler = MinMaxScaler()
    ur_values = scaler.fit_transform(np.array(ur_values).reshape(-1,1)).flatten()
    fe_values = scaler.fit_transform(np.array(fe_values).reshape(-1,1)).flatten()
    
    # Prior and Likelihood
    prior = ur_values
    likelihood = fe_values
    posterior = prior * likelihood
    posterior /= np.sum(posterior)  # Normalize to sum to 1
    return posterior

def platt_scaling(scores, labels):
    """
    Apply Platt scaling using logistic regression.
    """
    lr = LogisticRegression()
    lr.fit(scores.reshape(-1,1), labels)
    probs = lr.predict_proba(scores.reshape(-1,1))[:,1]
    return probs

def compute_final_bbox(boxes, weights):
    """
    Compute the final bounding box as a weighted average.
    """
    boxes = np.array(boxes)
    weights = np.array(weights)
    weights /= np.sum(weights)
    final_bbox = np.average(boxes, axis=0, weights=weights)
    return final_bbox

def load_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_image(image_folder, file_name):
    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {file_name} not found in {image_folder}")
    return image

def main():
    image_folder = 'test'
    json_file = 'full_anotation.json'
    data = load_json_file(json_file)
    for annotation in data['annotations']:
        image_info = next(img for img in data['images'] if img['id'] == annotation['image_id'])
        image = load_image(image_folder, image_info['file_name'])
        
        # Extract bounding boxes from the annotation
        boxes = [annotation[f'bbox{i}'] for i in range(1, 5)]
        # Example inputs
        user_ratings = [4.5, 4.0, 4.8, 2.0]  # Ratings from 0 to 5

        # Step 1: Remove outliers
        boxes_inliers = remove_outliers(boxes)
        print(f"Boxes after outlier removal: {boxes_inliers}")

        # Update user ratings and boxes to match inliers
        indices_inliers = [boxes.index(b) for b in boxes_inliers]
        user_ratings_inliers = [user_ratings[i] for i in indices_inliers]

        # Step 2: Compute IoU scores
        iou_scores = compute_iou_scores(boxes_inliers)
        print(f"IoU Scores: {iou_scores}")

        # Step 3: Extract features using CNN
        fe_values = extract_features(image, boxes_inliers)
        print(f"Feature Extraction Values: {fe_values}")

        # Step 4: Bayesian Update
        posterior_probs = bayesian_update(user_ratings_inliers, fe_values)
        print(f"Posterior Probabilities: {posterior_probs}")

        # Step 5: Platt Scaling
        # For demonstration, we'll create labels assuming higher posterior implies positive class
        labels = (posterior_probs > np.median(posterior_probs)).astype(int)
        combined_scores = iou_scores * posterior_probs
        platt_values = platt_scaling(combined_scores, labels)
        print(f"Platt Scaled Values: {platt_values}")

        # Step 6: Compute final bounding box
        final_bbox = compute_final_bbox(boxes_inliers, platt_values)
        print(f"Final Bounding Box: {final_bbox}")

        # === Visualization Code Starts Here ===
        # Draw user's bounding boxes (after outlier removal)
        # for box in boxes_inliers:
        #     x, y, w, h = map(int, box)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for user boxes

        # # Draw the final bounding box computed by the algorithm
        # x, y, w, h = map(int, final_bbox)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for final box

        # # Display the image with bounding boxes
        # cv2.imshow('Bounding Boxes', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # === Visualization Code Ends Here ===

if __name__ == "__main__":
    main()
