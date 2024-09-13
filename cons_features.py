import numpy as np
import cv2
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import json
import os

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
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  # Add epsilon to avoid division by zero
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
    Extract SIFT and HOG features for each bounding box.
    """
    sift = cv2.SIFT_create()
    # Define HOG Descriptor parameters
    win_size = (64, 128)  # HOG window size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    sift_values = []
    hog_values = []
    height, width = image.shape[:2]

    for box in boxes:
        x, y, w, h = map(int, box)
        # Ensure coordinates are within image boundaries
        x_end = min(x + w, width)
        y_end = min(y + h, height)
        x = max(x, 0)
        y = max(y, 0)
        w = x_end - x
        h = y_end - y
        roi = image[y:y_end, x:x_end]

        # Check if ROI is valid
        if roi.size == 0:
            print(f"Empty ROI for box: {box}")
            sift_values.append(0)
            hog_values.append(0)
            continue

        # Resize ROI to match HOG window size
        try:
            roi_resized = cv2.resize(roi, win_size)
        except cv2.error as e:
            print(f"Error resizing ROI for box {box}: {e}")
            sift_values.append(0)
            hog_values.append(0)
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        # SIFT
        kp, des = sift.detectAndCompute(gray, None)
        sift_value = len(kp) if kp is not None else 0
        sift_values.append(sift_value)

        # HOG
        try:
            hog_value = hog.compute(gray)
            hog_values.append(np.sum(hog_value))
        except cv2.error as e:
            print(f"Error computing HOG for box {box}: {e}")
            hog_values.append(0)
            continue

    # Normalize features
    sift_values = np.array(sift_values).reshape(-1, 1)
    hog_values = np.array(hog_values).reshape(-1, 1)
    scaler = MinMaxScaler()
    sift_values = scaler.fit_transform(sift_values).flatten()
    hog_values = scaler.fit_transform(hog_values).flatten()

    # Weighted average
    weights = [0.5, 0.5]  # Adjust weights as needed
    fe_values = weights[0]*sift_values + weights[1]*hog_values
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
    image_folder = 'test'  # Update this to your actual image folder
    json_file = 'full_annotation.json'  # Update this to your actual JSON file
    data = load_json_file(json_file)
    images = data['images']
    annotations = data['annotations']

    # For each image in the dataset
    for image_info in images:
        image_id = image_info['id']
        file_name = image_info['file_name']
        image = load_image(image_folder, file_name)

        # Get all annotations for this image
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # Extract bounding boxes from the annotations
        boxes = [ann['bbox'] for ann in image_annotations]

        # Check if there are bounding boxes
        if not boxes:
            print(f"No bounding boxes found for image {file_name}")
            continue

        # Simulate user ratings (assuming one per box)
        user_ratings = [4.5, 4.0, 4.8, 2.0, 4.6][:len(boxes)]  # Adjust length as needed

        # Ensure boxes and user_ratings have the same length
        if len(user_ratings) != len(boxes):
            user_ratings = user_ratings[:len(boxes)]

        # Step 1: Remove outliers
        boxes_inliers = remove_outliers(boxes)
        print(f"Boxes after outlier removal: {boxes_inliers}")

        # Update user ratings and boxes to match inliers
        indices_inliers = [boxes.index(b) for b in boxes_inliers]
        user_ratings_inliers = [user_ratings[i] for i in indices_inliers]

        # Step 2: Compute IoU scores
        iou_scores = compute_iou_scores(boxes_inliers)
        print(f"IoU Scores: {iou_scores}")

        # Step 3: Extract features
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
