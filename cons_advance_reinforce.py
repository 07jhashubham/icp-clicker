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
import random

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

class ExperienceReplayBuffer:
    """
    Experience Replay Buffer to store past experiences.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, reward):
        self.buffer.append((state, reward))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

def train_autoencoder_with_rl(autoencoder, roi_dataset, iou_scores, user_ratings, experience_replay, num_epochs=5):
    """
    Train the autoencoder using reinforcement learning principles with experience replay.
    """
    if torch.cuda.is_available():
        autoencoder = autoencoder.to('cuda')
        roi_dataset = roi_dataset.to('cuda')

    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
    gamma = 0.99  # Discount factor for future rewards
    batch_size = 8  # Batch size for experience replay
    epsilon = 0.1  # Exploration rate for epsilon-greedy strategy

    for epoch in range(num_epochs):
        autoencoder.train()
        total_reward = 0

        # Shuffle the dataset
        indices = np.arange(len(roi_dataset))
        np.random.shuffle(indices)

        for idx in indices:
            state = roi_dataset[idx]  # shape: [3, 224, 224]
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Exploration: Add random noise to the state
                action = state + torch.randn_like(state) * 0.1
                action = action.unsqueeze(0)  # Add batch dimension
            else:
                # Exploitation: Reconstruct the state
                action = autoencoder(state.unsqueeze(0))  # Add batch dimension

            # Compute dynamic reward
            reconstruction_error = F.mse_loss(action, state.unsqueeze(0))
            iou_score = iou_scores[idx] if iou_scores is not None else 0.0
            user_rating = user_ratings[idx] if user_ratings is not None else 2.5  # Neutral rating if not available

            # Normalize user rating
            user_rating_normalized = user_rating / 5.0  # Assuming ratings are from 0 to 5

            # Compute reward
            reward = -reconstruction_error.item() + iou_score + user_rating_normalized

            # Store experience
            experience_replay.push(state.detach(), reward)

            # Sample from experience replay
            experiences = experience_replay.sample(min(batch_size, len(experience_replay.buffer)))
            if len(experiences) == 0:
                continue  # Skip if no experiences to sample
            states_batch, rewards_batch = zip(*experiences)
            states_batch = torch.stack(states_batch)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
            if torch.cuda.is_available():
                rewards_batch = rewards_batch.to('cuda')
                states_batch = states_batch.to('cuda')

            # Compute predicted actions
            predicted_actions = autoencoder(states_batch)

            # Compute loss (negative expected reward)
            reconstruction_errors = F.mse_loss(predicted_actions, states_batch, reduction='none')
            reconstruction_errors = reconstruction_errors.view(reconstruction_errors.size(0), -1).mean(dim=1)
            loss = (reconstruction_errors - rewards_batch).mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward

        avg_reward = total_reward / len(roi_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Reward: {avg_reward:.4f}")

    autoencoder.eval()
    return autoencoder

def extract_features(image, boxes, autoencoder, experience_replay, retrain_interval=1, step_counter=0, iou_scores=None, user_ratings=None):
    """
    Extract features from each bounding box using an autoencoder with ResNet encoder.
    Periodically retrain the autoencoder using reinforcement learning with experience replay.
    Returns:
        fe_values: List of feature values.
        reconstruction_errors_normalized: Normalized reconstruction errors for each ROI.
        composite_confidence: Combined confidence scores incorporating IoU and reconstruction errors.
        autoencoder: The trained autoencoder.
        valid_indices: Indices of boxes for which features were extracted.
        step_counter: Updated step counter for retraining.
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
        return [], [], [], autoencoder, [], step_counter

    roi_dataset = torch.stack(roi_tensors)

    # Periodically retrain the autoencoder using RL
    if step_counter % retrain_interval == 0:
        print("Retraining autoencoder using reinforcement learning...")
        autoencoder = train_autoencoder_with_rl(autoencoder, roi_dataset, iou_scores, user_ratings, experience_replay)
    step_counter += 1

    # Compute reconstruction errors
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
    recon_errors_normalized = MinMaxScaler().fit_transform(recon_errors.reshape(-1, 1)).flatten()

    # Combine IoU scores with Reconstruction Errors to form a composite confidence score
    if iou_scores is not None:
        # Select IoU scores corresponding to valid_indices
        iou_scores_valid = np.array(iou_scores)[valid_indices]
        # Ensure iou_scores_valid has the same length as recon_errors_normalized
        if len(iou_scores_valid) != len(recon_errors_normalized):
            print("Mismatch in lengths between IoU scores and reconstruction errors. Adjusting...")
            min_length = min(len(iou_scores_valid), len(recon_errors_normalized))
            iou_scores_valid = iou_scores_valid[:min_length]
            recon_errors_normalized = recon_errors_normalized[:min_length]
            fe_values = fe_values[:min_length]
            valid_indices = valid_indices[:min_length]
        composite_confidence = 0.7 * iou_scores_valid + 0.3 * (1 - recon_errors_normalized)
    else:
        # If IoU scores are not available, rely solely on reconstruction errors
        composite_confidence = 1 - recon_errors_normalized

    # Debug statements to verify lengths
    print(f"Length of iou_scores_valid: {len(iou_scores_valid)}")
    print(f"Length of recon_errors_normalized: {len(recon_errors_normalized)}")
    print(f"Length of composite_confidence: {len(composite_confidence)}")
    print(f"Length of valid_indices: {len(valid_indices)}")

    return fe_values, recon_errors_normalized, composite_confidence, autoencoder, valid_indices, step_counter

def bayesian_update(user_ratings, composite_confidence):
    """
    Compute posterior probabilities using Bayesian update by integrating user ratings and composite confidence.
    """
    epsilon = 1e-6  # Small value to avoid zeros

    # Normalize user ratings
    user_ratings = np.array(user_ratings).reshape(-1)
    if np.max(user_ratings) == np.min(user_ratings):
        user_ratings_normalized = np.full(user_ratings.shape, 0.5)
    else:
        scaler = MinMaxScaler()
        user_ratings_normalized = scaler.fit_transform(user_ratings.reshape(-1, 1)).flatten()

    # Normalize composite confidence
    composite_confidence = np.array(composite_confidence).reshape(-1)
    if np.max(composite_confidence) == np.min(composite_confidence):
        composite_confidence_normalized = np.full(composite_confidence.shape, 0.5)
    else:
        scaler = MinMaxScaler()
        composite_confidence_normalized = scaler.fit_transform(composite_confidence.reshape(-1, 1)).flatten()

    # Add epsilon to avoid zeros
    user_ratings_normalized = user_ratings_normalized * (1 - 2 * epsilon) + epsilon
    composite_confidence_normalized = composite_confidence_normalized * (1 - 2 * epsilon) + epsilon

    # Prior and Likelihood
    prior = user_ratings_normalized
    likelihood = composite_confidence_normalized
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

def compute_final_bbox(boxes, weights, recon_errors_normalized):
    """
    Compute the final bounding box as a weighted average, incorporating reconstruction errors.
    """
    boxes = np.array(boxes)
    weights = np.array(weights)
    # Incorporate reconstruction errors: higher errors reduce the weight
    adjusted_weights = weights * (1 - recon_errors_normalized)
    # Normalize the adjusted weights
    if np.sum(adjusted_weights) == 0:
        adjusted_weights = np.full_like(adjusted_weights, 1.0 / len(adjusted_weights))
    else:
        adjusted_weights /= np.sum(adjusted_weights)
    final_bbox = np.average(boxes, axis=0, weights=adjusted_weights)
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
    json_file = '/media/joeru/380C4A280C49E18C/projects/akaispace-consensus/icp-clicker/full_anotation.json'
    data = load_json_file(json_file)
    autoencoder = ResNetAutoencoder()  # Initialize autoencoder outside the loop for reuse
    experience_replay = ExperienceReplayBuffer(capacity=50)
    step_counter = 0  # Counter to track steps for periodic retraining

    for annotation in data['annotations']:
        image_info = next(img for img in data['images'] if img['id'] == annotation['image_id'])
        image = load_image(image_folder, image_info['file_name'])

        # Extract bounding boxes from the annotation
        boxes = [annotation[f'bbox{i}'] for i in range(1, 5)]
        # User ratings (if available); for active learning, we may not have these initially
        user_ratings = [2.5, 2.5, 2.5, 2.5]  # Ratings from 0 to 5

        # Step 1: Remove outliers
        boxes_inliers = remove_outliers(boxes)
        print(f"Boxes after outlier removal: {boxes_inliers}")

        # Update user ratings to match inliers
        indices_inliers = [boxes.index(b) for b in boxes_inliers]
        user_ratings_inliers = [user_ratings[i] for i in indices_inliers]

        # Step 2: Compute IoU scores
        iou_scores = compute_iou_scores(boxes_inliers)
        print(f"IoU Scores: {iou_scores}")

        # Step 3: Extract features and compute composite confidence
        fe_values, recon_errors_normalized, composite_confidence, autoencoder, valid_indices, step_counter = extract_features(
            image, boxes_inliers, autoencoder, experience_replay, retrain_interval=2, step_counter=step_counter,
            iou_scores=iou_scores, user_ratings=user_ratings_inliers
        )
        print(f"Feature Values: {fe_values}")
        print(f"Reconstruction Errors (Normalized): {recon_errors_normalized}")
        print(f"Composite Confidence: {composite_confidence}")

        # Update user ratings, boxes, and iou_scores to match valid_indices
        user_ratings_valid = [user_ratings_inliers[i] for i in valid_indices]
        boxes_valid = [boxes_inliers[i] for i in valid_indices]
        iou_scores_valid = [iou_scores[i] for i in valid_indices]

        # Active Learning Decision
        # If user ratings are not available, use composite confidence to simulate user ratings
        if not user_ratings_valid:
            print("No user ratings available. Assigning simulated ratings based on composite confidence.")
            # Simulate user ratings: Higher composite confidence leads to higher simulated ratings
            composite_confidence_np = np.array(composite_confidence)
            simulated_ratings = 5.0 * composite_confidence_np  # Scale to 0-5
            user_ratings_valid = simulated_ratings.tolist()

        # Ensure lengths match
        if len(composite_confidence) != len(user_ratings_valid):
            print("Mismatch in lengths after active learning. Adjusting...")
            min_length = min(len(composite_confidence), len(user_ratings_valid))
            composite_confidence = composite_confidence[:min_length]
            user_ratings_valid = user_ratings_valid[:min_length]
            iou_scores_valid = iou_scores_valid[:min_length]
            boxes_valid = boxes_valid[:min_length]

        # Step 4: Bayesian Update using composite confidence
        posterior_probs = bayesian_update(user_ratings_valid, composite_confidence)
        print(f"Posterior Probabilities: {posterior_probs}")

        # Step 5: Platt Scaling
        labels = (posterior_probs > np.median(posterior_probs)).astype(int)
        combined_scores = np.array(iou_scores_valid) * posterior_probs
        platt_values = platt_scaling(combined_scores, labels)
        print(f"Platt Scaled Values: {platt_values}")

        # Step 6: Compute final bounding box
        final_bbox = compute_final_bbox(boxes_valid, platt_values, recon_errors_normalized)  # Updated call
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
