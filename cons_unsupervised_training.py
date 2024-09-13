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
import torch.nn as nn
import torch.optim as optim
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
        iou_value = interArea / float(boxAArea + boxBArea - interArea)
        return iou_value

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

class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet50(pretrained=True)
        # Encoder: Use layers up to the second last convolutional block
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Exclude avgpool and fc layers

        # Decoder: Corrected to output 224x224 images
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # Output: [batch_size, 1024, 14, 14]
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),   # Output: [batch_size, 512, 28, 28]
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # Output: [batch_size, 256, 56, 56]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # Output: [batch_size, 128, 112, 112]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # Output: [batch_size, 64, 224, 224]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),       # Output: [batch_size, 3, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def extract_features(image, boxes, autoencoder=None):
    """
    Extract features from each bounding box using an autoencoder with ResNet encoder.
    Returns:
        fe_values: List of feature values.
        autoencoder: The trained autoencoder.
        valid_indices: Indices of boxes for which features were extracted.
    """
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    features_list = []
    roi_tensors = []
    valid_indices = []
    height, width = image.shape[:2]

    for idx, box in enumerate(boxes):
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
            continue

        # Preprocess ROI
        input_tensor = preprocess(roi)
        roi_tensors.append(input_tensor)
        valid_indices.append(idx)

    # Stack all ROI tensors
    if len(roi_tensors) == 0:
        return [], autoencoder, []

    roi_dataset = torch.stack(roi_tensors)

    # Train autoencoder if not provided
    if autoencoder is None:
        autoencoder = ResNetAutoencoder()
        if torch.cuda.is_available():
            autoencoder = autoencoder.to('cuda')
            roi_dataset = roi_dataset.to('cuda')
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
        num_epochs = 5  # Adjust as needed

        # Train the autoencoder
        autoencoder.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = autoencoder(roi_dataset)
            loss = criterion(outputs, roi_dataset)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        autoencoder.eval()

    # Extract features using the encoder part
    with torch.no_grad():
        if torch.cuda.is_available():
            roi_dataset = roi_dataset.to('cuda')
        features = autoencoder.encoder(roi_dataset)
        features = features.cpu().numpy()
        # Flatten the feature maps
        features = features.reshape(features.shape[0], -1)

    # For simplicity, compute the L2 norm of each feature vector
    fe_values = [np.linalg.norm(f) for f in features]
    # Normalize features
    scaler = MinMaxScaler()
    fe_values = scaler.fit_transform(np.array(fe_values).reshape(-1, 1)).flatten()
    return fe_values, autoencoder, valid_indices  # Return valid indices

def bayesian_update(ur_values, fe_values):
    """
    Compute posterior probabilities using Bayesian update.
    """
    epsilon = 1e-6  # Small value to avoid zeros

    # Normalize UR and FE values
    ur_values = np.array(ur_values).reshape(-1)
    fe_values = np.array(fe_values).reshape(-1)

    # Handle case when all values are the same
    if np.max(ur_values) == np.min(ur_values):
        ur_values_normalized = np.full(ur_values.shape, 0.5)
    else:
        scaler = MinMaxScaler()
        ur_values_normalized = scaler.fit_transform(ur_values.reshape(-1, 1)).flatten()

    if np.max(fe_values) == np.min(fe_values):
        fe_values_normalized = np.full(fe_values.shape, 0.5)
    else:
        scaler = MinMaxScaler()
        fe_values_normalized = scaler.fit_transform(fe_values.reshape(-1, 1)).flatten()

    # Add epsilon to avoid zeros
    ur_values_normalized = ur_values_normalized * (1 - 2 * epsilon) + epsilon
    fe_values_normalized = fe_values_normalized * (1 - 2 * epsilon) + epsilon

    # Prior and Likelihood
    prior = ur_values_normalized
    likelihood = fe_values_normalized
    posterior = prior * likelihood
    sum_posterior = np.sum(posterior)

    if sum_posterior == 0 or np.isnan(sum_posterior):
        # Assign equal probability if sum is zero or NaN
        posterior = np.full_like(posterior, 1.0 / len(posterior))
    else:
        posterior /= sum_posterior  # Normalize to sum to 1

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
        user_ratings = [2.5,2.5,2.5,2.5,]  # Ratings from 0 to 5

        # Step 1: Remove outliers
        boxes_inliers = remove_outliers(boxes)
        print(f"Boxes after outlier removal: {boxes_inliers}")

        # Update user ratings to match inliers
        indices_inliers = [boxes.index(b) for b in boxes_inliers]
        user_ratings_inliers = [user_ratings[i] for i in indices_inliers]

        # Step 2: Compute IoU scores
        iou_scores = compute_iou_scores(boxes_inliers)
        print(f"IoU Scores: {iou_scores}")

        # Step 3: Extract features using Autoencoder with ResNet encoder
        fe_values, autoencoder, valid_indices = extract_features(image, boxes_inliers)
        print(f"Feature Extraction Values: {fe_values}")

        # Update user ratings, boxes, and iou_scores to match valid_indices
        user_ratings_valid = [user_ratings_inliers[i] for i in valid_indices]
        boxes_valid = [boxes_inliers[i] for i in valid_indices]
        iou_scores_valid = [iou_scores[i] for i in valid_indices]

        # Check if we have valid data
        if len(fe_values) == 0 or len(user_ratings_valid) == 0:
            print("No valid data available for further processing.")
            continue

        # Step 4: Bayesian Update
        posterior_probs = bayesian_update(user_ratings_valid, fe_values)
        print(f"Posterior Probabilities: {posterior_probs}")

        # Step 5: Platt Scaling
        labels = (posterior_probs > np.median(posterior_probs)).astype(int)
        combined_scores = np.array(iou_scores_valid) * posterior_probs
        platt_values = platt_scaling(combined_scores, labels)
        print(f"Platt Scaled Values: {platt_values}")

        # Step 6: Compute final bounding box
        final_bbox = compute_final_bbox(boxes_valid, platt_values)
        print(f"Final Bounding Box: {final_bbox}")

        # === Visualization Code (optional) ===
        # Adjust visualization to use boxes_valid
        # for box in boxes_valid:
        #     x, y, w, h = map(int, box)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # x, y, w, h = map(int, final_bbox)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.imshow('Bounding Boxes', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # === Visualization Code Ends Here ===

if __name__ == "__main__":
    main()
