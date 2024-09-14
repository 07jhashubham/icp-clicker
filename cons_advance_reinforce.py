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
from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
# Initialize the tokenizer globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)

def remove_outliers(boxes):
    """
    Remove outlier bounding boxes using DBSCAN clustering.
    """
    data = np.array(boxes)
    # Normalize the data to ensure equal weighting
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(data_scaled)
    labels = clustering.labels_

    # Identify the largest cluster (excluding noise labeled as -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        print("No clusters found. Returning all boxes as inliers.")
        return boxes

    largest_cluster_label = unique_labels[np.argmax(counts)]
    inliers = [boxes[i] for i in range(len(boxes)) if labels[i] == largest_cluster_label]

    return inliers

from scipy.optimize import minimize

def geometric_median(points, eps=1e-5):
    """
    Compute the geometric median of a set of points.
    """
    points = np.asarray(points)

    def aggregate_distance(x):
        return np.sum(np.linalg.norm(points - x, axis=1))

    centroid = np.mean(points, axis=0)
    result = minimize(aggregate_distance, centroid, method='BFGS')
    return result.x

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
    def __init__(self, text_embedding_dim=768):
        super(ResNetAutoencoder, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # Encoder: Use layers up to the second last convolutional block
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Exclude avgpool and fc layers

        # Text embedding using pre-trained BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        self.attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8)
        # Adjust the decoder input channels to accommodate combined features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048 + text_embedding_dim, 1024, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
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

    def forward(self, image_input, text_input_ids, text_attention_mask):
        # Encode the image
        encoded_image = self.encoder(image_input)  # Shape: [batch_size, 2048, 7, 7]

        # Process text input through BERT
        bert_outputs = self.bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embedding = bert_outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation

        # Expand text_embedding to match spatial dimensions
        text_embedding = text_embedding.unsqueeze(2).unsqueeze(3)  # Shape: [batch_size, text_embedding_dim, 1, 1]
        text_embedding_expanded = text_embedding.expand(-1, -1, encoded_image.size(2), encoded_image.size(3))  # Shape: [batch_size, text_embedding_dim, 7, 7]

        # Concatenate along the channel dimension
        combined_features = torch.cat([encoded_image, text_embedding_expanded], dim=1)  # Shape: [batch_size, 2048 + text_embedding_dim, 7, 7]

        # Decode the combined features
        decoded_output = self.decoder(combined_features)
        return decoded_output


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer to store past experiences.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state_tuple, reward):
        self.buffer.append((state_tuple, reward))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))


def train_autoencoder_with_rl(autoencoder, roi_dataset, text_inputs, iou_scores, user_ratings, experience_replay, num_epochs=5):
    """
    Train the autoencoder using reinforcement learning principles with experience replay.
    """
    if torch.cuda.is_available():
        autoencoder = autoencoder.to('cuda')
        roi_dataset = roi_dataset.to('cuda')
        text_input_ids = text_inputs['input_ids'].to('cuda')
        text_attention_mask = text_inputs['attention_mask'].to('cuda')
    else:
        text_input_ids = text_inputs['input_ids']
        text_attention_mask = text_inputs['attention_mask']

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
            text_input_id = text_input_ids[idx]
            text_attention = text_attention_mask[idx]

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Exploration: Add random noise to the state
                action = state + torch.randn_like(state) * 0.1
                action = action.unsqueeze(0)  # Add batch dimension
            else:
                # Exploitation: Reconstruct the state
                action = autoencoder(state.unsqueeze(0), text_input_id.unsqueeze(0), text_attention.unsqueeze(0))  # Add batch dimension

            # Compute dynamic reward
            reconstruction_error = F.mse_loss(action, state.unsqueeze(0))
            iou_score = iou_scores[idx] if iou_scores is not None else 0.0
            user_rating = user_ratings[idx] if user_ratings is not None else 2.5  # Neutral rating if not available

            # Normalize user rating
            user_rating_normalized = user_rating / 5.0  # Assuming ratings are from 0 to 5

            # Compute reward
            reward = -reconstruction_error.item() + iou_score + user_rating_normalized

            # Store experience
            experience_replay.push((state.detach(), text_input_id.detach(), text_attention.detach()), reward)

            # Sample from experience replay
            experiences = experience_replay.sample(min(batch_size, len(experience_replay.buffer)))
            if len(experiences) == 0:
                continue  # Skip if no experiences to sample
            states_batch, rewards_batch = zip(*experiences)
            image_states_batch, text_input_ids_batch, text_attention_masks_batch = zip(*[s for s in states_batch])
            image_states_batch = torch.stack(image_states_batch)
            text_input_ids_batch = torch.stack(text_input_ids_batch)
            text_attention_masks_batch = torch.stack(text_attention_masks_batch)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
            if torch.cuda.is_available():
                rewards_batch = rewards_batch.to('cuda')
                image_states_batch = image_states_batch.to('cuda')
                text_input_ids_batch = text_input_ids_batch.to('cuda')
                text_attention_masks_batch = text_attention_masks_batch.to('cuda')

            # Compute predicted actions
            predicted_actions = autoencoder(image_states_batch, text_input_ids_batch, text_attention_masks_batch)

            # Compute loss (negative expected reward)
            reconstruction_errors = F.mse_loss(predicted_actions, image_states_batch, reduction='none')
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


def extract_features(image, boxes, autoencoder, experience_replay, retrain_interval=1, step_counter=0, iou_scores=None, user_ratings=None, text_inputs=None):
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
    text_input_ids_list = []
    text_attention_masks_list = []
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

        # Get the corresponding text input
        text_input = text_inputs[idx]
        # Tokenize the text
        encoding = tokenizer(
            text_input,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
            clean_up_tokenization_spaces=False  # Suppress FutureWarning
        )
        text_input_ids_list.append(encoding['input_ids'].squeeze(0))
        text_attention_masks_list.append(encoding['attention_mask'].squeeze(0))

    # Stack all ROI tensors
    if len(roi_tensors) == 0:
        return [], [], [], autoencoder, [], step_counter

    roi_dataset = torch.stack(roi_tensors)
    text_input_ids = torch.stack(text_input_ids_list)
    text_attention_masks = torch.stack(text_attention_masks_list)

    # Periodically retrain the autoencoder using RL
    if step_counter % retrain_interval == 0:
        print("Retraining autoencoder using reinforcement learning...")
        autoencoder = train_autoencoder_with_rl(
            autoencoder,
            roi_dataset,
            {'input_ids': text_input_ids, 'attention_mask': text_attention_masks},
            iou_scores,
            user_ratings,
            experience_replay
        )
    step_counter += 1

    # Compute reconstruction errors
    if torch.cuda.is_available():
        autoencoder = autoencoder.to('cuda')
        roi_dataset = roi_dataset.to('cuda')
        text_input_ids = text_input_ids.to('cuda')
        text_attention_masks = text_attention_masks.to('cuda')
    with torch.no_grad():
        outputs = autoencoder(roi_dataset, text_input_ids, text_attention_masks)
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

# def compute_final_bbox(boxes, weights, recon_errors_normalized):
#     """
#     Compute the final bounding box as a weighted average, incorporating reconstruction errors.
#     """
#     boxes = np.array(boxes)
#     weights = np.array(weights)
#     # Incorporate reconstruction errors: higher errors reduce the weight
#     adjusted_weights = weights * (1 - recon_errors_normalized)
#     # Normalize the adjusted weights
#     if np.sum(adjusted_weights) == 0:
#         adjusted_weights = np.full_like(adjusted_weights, 1.0 / len(adjusted_weights))
#     else:
#         adjusted_weights /= np.sum(adjusted_weights)
#     final_bbox = np.average(boxes, axis=0, weights=adjusted_weights)
#     return final_bbox

from ensemble_boxes import weighted_boxes_fusion
def compute_final_bbox(boxes, weights, recon_errors_normalized, image_width, image_height):
    """
    Compute the final bounding box using Weighted Box Fusion (WBF), incorporating reconstruction errors
    and geometric median.
    
    Args:
        boxes: List of boxes in [x, y, w, h] format.
        recon_errors_normalized: Normalized reconstruction errors (array-like).
        image_width: Width of the image.
        image_height: Height of the image.
    
    Returns:
        final_bbox: The fused bounding box in [x, y, w, h] format.
    """
    # Convert boxes to [x1, y1, x2, y2] format
    boxes_xyxy = []
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        boxes_xyxy.append([x1, y1, x2, y2])
    boxes_xyxy = np.array(boxes_xyxy)
    
    # Normalize boxes to [0, 1]
    boxes_norm = boxes_xyxy.copy()
    boxes_norm[:, 0] /= image_width
    boxes_norm[:, 1] /= image_height
    boxes_norm[:, 2] /= image_width
    boxes_norm[:, 3] /= image_height
    
    # Compute geometric median
    median_box = geometric_median(boxes_xyxy)
    # Normalize geometric median box
    median_box_norm = median_box.copy()
    median_box_norm[0] /= image_width
    median_box_norm[1] /= image_height
    median_box_norm[2] /= image_width
    median_box_norm[3] /= image_height
    
    # Prepare boxes_list, scores_list, labels_list for WBF
    # Each user's boxes are treated as from one 'detector'
    boxes_list = [boxes_norm.tolist(), [median_box_norm.tolist()]]  # Add median box as an additional detector
    # Compute confidence scores from reconstruction errors
    scores = weights * (1 - recon_errors_normalized)
    # Normalize the adjusted weights
    if np.sum(scores) == 0:
        scores = np.full_like(scores, 1.0 / len(scores))
    else:
        scores /= np.sum(scores)
    scores_median = [1.0]  # Assign high confidence to the geometric median box
    scores_list = [scores, scores_median]
    labels_list = [[0] * len(boxes), [0]]  # Assuming single-class detection
    
    # Assign weights to detectors (optional)
    weights = [1.0, 2.0]  # Give higher weight to the geometric median box
    
    # Apply WBF
    boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0, weights=weights)
    
    # Denormalize boxes back to original image dimensions
    boxes_fused = np.array(boxes_fused)
    boxes_fused[:, 0] *= image_width
    boxes_fused[:, 1] *= image_height
    boxes_fused[:, 2] *= image_width
    boxes_fused[:, 3] *= image_height
    
    # Since WBF may return multiple boxes, we'll take the one with the highest score
    max_score_idx = np.argmax(scores_fused)
    fused_box = boxes_fused[max_score_idx]
    
    # Convert back to [x, y, w, h] format
    x = fused_box[0]
    y = fused_box[1]
    w = fused_box[2] - fused_box[0]
    h = fused_box[3] - fused_box[1]
    
    final_bbox = [x, y, w, h]
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

        # Extract text inputs for each box
        # Assuming annotation contains text data per box in a list 'texts'
        text_inputs = annotation.get('texts', ["{Objective: To draw the bounding boxes around the car's number plate}"] * len(boxes))  # Empty string if not available

        # Step 1: Remove outliers
        boxes_inliers = remove_outliers(boxes)
        print(f"Boxes after outlier removal: {boxes_inliers}")

        # Update user ratings and text_inputs to match inliers
        indices_inliers = [boxes.index(b) for b in boxes_inliers]
        user_ratings_inliers = [user_ratings[i] for i in indices_inliers]
        text_inputs_inliers = [text_inputs[i] for i in indices_inliers]

        # Step 2: Compute IoU scores
        iou_scores = compute_iou_scores(boxes_inliers)
        print(f"IoU Scores: {iou_scores}")

        # Step 3: Extract features and compute composite confidence
        fe_values, recon_errors_normalized, composite_confidence, autoencoder, valid_indices, step_counter = extract_features(
            image, boxes_inliers, autoencoder, experience_replay, retrain_interval=2, step_counter=step_counter,
            iou_scores=iou_scores, user_ratings=user_ratings_inliers, text_inputs=text_inputs_inliers
        )
        print(f"Feature Values: {fe_values}")
        print(f"Reconstruction Errors (Normalized): {recon_errors_normalized}")
        print(f"Composite Confidence: {composite_confidence}")

        # Update user ratings, boxes, iou_scores, and text_inputs to match valid_indices
        user_ratings_valid = [user_ratings_inliers[i] for i in valid_indices]
        boxes_valid = [boxes_inliers[i] for i in valid_indices]
        iou_scores_valid = [iou_scores[i] for i in valid_indices]
        text_inputs_valid = [text_inputs_inliers[i] for i in valid_indices]

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
        final_bbox = compute_final_bbox(boxes_valid, platt_values, recon_errors_normalized, image_height=image.shape[1], image_width=image.shape[0])
        print(f"Final Bounding Box: {final_bbox}")

        visualize_bounding_boxes(
            image=image,
            user_boxes=boxes_valid,
            predicted_box=final_bbox,
            output_path="visualized_output",
            image_name=image_info['file_name']
        )


if __name__ == "__main__":
    main()
