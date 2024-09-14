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
        resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # Encoder: Use layers up to the second last convolutional block
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Exclude avgpool and fc layers

        # Decoder: Corrected to output 224x224 images
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # 28x28 -> 56x56
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # 56x56 -> 112x112
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # 112x112 -> 224x224
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),       # 224x224 -> 224x224
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
        reconstruction_errors: Reconstruction errors for each ROI.
        autoencoder: The trained autoencoder.
        valid_indices: Indices of boxes for which features were extracted.
    """
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

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
        return [], [], autoencoder, []

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

        # Active Learning Loop
        for epoch in range(num_epochs):
            autoencoder.train()
            optimizer.zero_grad()
            outputs = autoencoder(roi_dataset)
            loss = criterion(outputs, roi_dataset)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Uncertainty Estimation
            reconstruction_errors = F.mse_loss(outputs, roi_dataset, reduction='none')
            reconstruction_errors = reconstruction_errors.view(reconstruction_errors.size(0), -1).mean(dim=1)
            # Detach reconstruction errors
            reconstruction_errors = reconstruction_errors.detach()
            uncertainty_threshold = reconstruction_errors.mean().item()

            # Select uncertain samples for further training
            uncertain_indices = (reconstruction_errors > uncertainty_threshold).nonzero(as_tuple=True)[0]
            if len(uncertain_indices) == 0:
                break  # Exit if no uncertain samples

            # Additional training on uncertain samples
            optimizer.zero_grad()
            uncertain_data = roi_dataset[uncertain_indices]
            outputs = autoencoder(uncertain_data)
            loss = criterion(outputs, uncertain_data)
            loss.backward()
            optimizer.step()
        autoencoder.eval()
    else:
        # If autoencoder is provided, just compute reconstruction errors
        if torch.cuda.is_available():
            autoencoder = autoencoder.to('cuda')
            roi_dataset = roi_dataset.to('cuda')
        with torch.no_grad():
            outputs = autoencoder(roi_dataset)
            reconstruction_errors = F.mse_loss(outputs, roi_dataset, reduction='none')
            reconstruction_errors = reconstruction_errors.view(reconstruction_errors.size(0), -1).mean(dim=1)
            # Detach reconstruction errors
            reconstruction_errors = reconstruction_errors.detach()

    # Extract features using the encoder part
    with torch.no_grad():
        features = autoencoder.encoder(roi_dataset)
        features = features.cpu().numpy()
        # Flatten the feature maps
        features = features.reshape(features.shape[0], -1)

    # For simplicity, compute the L2 norm of each feature vector
    fe_values = [np.linalg.norm(f) for f in features]
    # Normalize features
    scaler = MinMaxScaler()
    fe_values = scaler.fit_transform(np.array(fe_values).reshape(-1, 1)).flatten()

    # Normalize reconstruction errors
    recon_errors = reconstruction_errors.cpu().numpy()
    recon_errors = MinMaxScaler().fit_transform(recon_errors.reshape(-1, 1)).flatten()

    return fe_values, recon_errors, autoencoder, valid_indices  # Return reconstruction errors and valid indices

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
def visualize_bounding_boxes(image, user_boxes, predicted_box, output_path, image_name):
    """
    Draws user-submitted and predicted bounding boxes on the image and saves it.

    Parameters:
        image (numpy.ndarray): The original image.
        user_boxes (list): List of user-submitted bounding boxes [x, y, w, h].
        predicted_box (list or numpy.ndarray): The predicted bounding box [x, y, w, h].
        output_path (str): Directory where the visualized image will be saved.
        image_name (str): The name of the image file.
    """
    # Create a copy of the image to draw on
    vis_image = image.copy()

    # Define colors (BGR format)
    user_box_color = (255, 0, 0)       # Blue for user boxes
    predicted_box_color = (0, 0, 255)  # Red for predicted box

    # Define thickness
    thickness = 2

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Draw user-submitted bounding boxes
    for idx, box in enumerate(user_boxes):
        x, y, w, h = map(int, box)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(vis_image, top_left, bottom_right, user_box_color, thickness)
        label = f'User Box {idx+1}'
        cv2.putText(vis_image, label, (x, y - 10), font, font_scale, user_box_color, font_thickness, cv2.LINE_AA)

    # Draw predicted bounding box
    if predicted_box is not None:
        x, y, w, h = map(int, predicted_box)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(vis_image, top_left, bottom_right, predicted_box_color, thickness)
        label = 'Predicted Box'
        cv2.putText(vis_image, label, (x, y - 10), font, font_scale, predicted_box_color, font_thickness, cv2.LINE_AA)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the visualized image
    save_path = os.path.join(output_path, image_name)
    cv2.imwrite(save_path, vis_image)
    print(f"Saved visualized image to {save_path}")

def main():
    image_folder = 'test'
    json_file = 'full_anotation.json'
    data = load_json_file(json_file)
    autoencoder = None  # Initialize autoencoder outside the loop for reuse

    for annotation in data['annotations']:
        image_info = next(img for img in data['images'] if img['id'] == annotation['image_id'])
        image = load_image(image_folder, image_info['file_name'])

        # Extract bounding boxes from the annotation
        boxes = [annotation[f'bbox{i}'] for i in range(1, 5)]
        # User ratings (if available); for active learning, we may not have these initially
        user_ratings = [2.5,2.5,2.5,2.5]  # Ratings from 0 to 5

        # Step 1: Remove outliers
        boxes_inliers = remove_outliers(boxes)
        print(f"Boxes after outlier removal: {boxes_inliers}")

        # Update user ratings to match inliers
        indices_inliers = [boxes.index(b) for b in boxes_inliers]
        user_ratings_inliers = [user_ratings[i] for i in indices_inliers]

        # Step 2: Compute IoU scores
        iou_scores = compute_iou_scores(boxes_inliers)
        print(f"IoU Scores: {iou_scores}")

        # Step 3: Extract features and reconstruction errors
        fe_values, recon_errors, autoencoder, valid_indices = extract_features(image, boxes_inliers, autoencoder)
        print(f"Feature Values: {fe_values}")
        print(f"Reconstruction Errors: {recon_errors}")

        # Update user ratings, boxes, and iou_scores to match valid_indices
        user_ratings_valid = [user_ratings_inliers[i] for i in valid_indices]
        boxes_valid = [boxes_inliers[i] for i in valid_indices]
        iou_scores_valid = [iou_scores[i] for i in valid_indices]

        # Active Learning Decision
        # If user ratings are not available, use reconstruction errors as uncertainty measure
        if not user_ratings_valid:
            print("No user ratings available. Assigning default ratings.")
            # Simulate user ratings without prompting
            # Assign higher ratings to samples with lower reconstruction errors (simulating user preference)
            recon_errors_np = np.array(recon_errors)
            # Invert reconstruction errors to simulate ratings (lower error -> higher rating)
            simulated_ratings = 5.0 * (1 - recon_errors_np / recon_errors_np.max())
            user_ratings_valid = simulated_ratings.tolist()

        # Ensure lengths match
        if len(fe_values) != len(user_ratings_valid):
            print("Mismatch in lengths after active learning. Adjusting...")
            min_length = min(len(fe_values), len(user_ratings_valid))
            fe_values = fe_values[:min_length]
            user_ratings_valid = user_ratings_valid[:min_length]
            iou_scores_valid = iou_scores_valid[:min_length]
            boxes_valid = boxes_valid[:min_length]

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

        visualize_bounding_boxes(
            image=image,
            user_boxes=boxes_valid,
            predicted_box=final_bbox,
            output_path="visualized_output_unsupervised",
            image_name=image_info['file_name']
        )
if __name__ == "__main__":
    main()