import numpy as np

# Function to calculate IoU between two bounding boxes
def calculate_iou(boxA, boxB):
    # Coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to calculate weighted average of bounding boxes
def weighted_average_bbox(boxes, weights):
    total_weight = sum(weights)
    x_avg = sum([w * box[0] for box, w in zip(boxes, weights)]) / total_weight
    y_avg = sum([w * box[1] for box, w in zip(boxes, weights)]) / total_weight
    width_avg = sum([w * box[2] for box, w in zip(boxes, weights)]) / total_weight
    height_avg = sum([w * box[3] for box, w in zip(boxes, weights)]) / total_weight
    return [x_avg, y_avg, width_avg, height_avg]

# Function to find clusters of bounding boxes using IoU threshold
def cluster_bboxes(bboxes, iou_threshold=0.5):
    clusters = []
    for box in bboxes:
        added = False
        for cluster in clusters:
            if calculate_iou(cluster[0], box) > iou_threshold:
                cluster.append(box)
                added = True
                break
        if not added:
            clusters.append([box])
    return clusters

# Example bounding boxes from users: [x, y, width, height]
bboxes = [
    [50, 50, 100, 100],  # User 1
    [52, 48, 102, 98],   # User 2
    [200, 200, 100, 100] # User 3 (outlier)
]

# Example user weights (based on their performance)
weights = [0.8, 0.9, 0.5]

# Step 1: Cluster bounding boxes based on IoU
clusters = cluster_bboxes(bboxes, iou_threshold=0.5)

# Step 2: Calculate consensus bounding box for each cluster
consensus_bboxes = []
for cluster in clusters:
    # Get the weights for the users in the current cluster
    cluster_weights = [weights[bboxes.index(box)] for box in cluster]
    # Calculate the weighted average bounding box for this cluster
    consensus_box = weighted_average_bbox(cluster, cluster_weights)
    consensus_bboxes.append(consensus_box)

# Output final consensus bounding boxes
print("Consensus Bounding Boxes:", consensus_bboxes)
