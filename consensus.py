import json
import numpy as np

# Function to calculate IoU between two bounding boxes
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

# Function to process annotations and find consensus bounding box
def process_annotations(data, iou_threshold=0.5):
    for annotation in data["annotations"]:
        # Extract bounding boxes and areas
        bboxes = [annotation[f"bbox{i}"] for i in range(1, 5)]
        areas = [annotation[f"area{i}"] for i in range(1, 5)]
        
        # Example user weights (these can be dynamic based on user performance)
        weights = [1 / area for area in areas]  # You can adjust this weight calculation based on your scoring method
        
        # Step 1: Cluster bounding boxes based on IoU
        clusters = cluster_bboxes(bboxes, iou_threshold=iou_threshold)
        
        # Step 2: Calculate consensus bounding box for each cluster
        consensus_bboxes = []
        for cluster in clusters:
            cluster_weights = [weights[bboxes.index(box)] for box in cluster]
            consensus_box = weighted_average_bbox(cluster, cluster_weights)
            consensus_bboxes.append(consensus_box)
        
        # Step 3: Assign final consensus to the annotation
        if consensus_bboxes:
            annotation["final_bbox"] = consensus_bboxes[0]  # You can handle multiple clusters differently
        else:
            annotation["final_bbox"] = None

    return data

# Function to read JSON file, process, and write updated annotations back
def update_annotations_with_consensus(file_name):
    # Load the JSON file
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    # Process annotations
    updated_data = process_annotations(data)

    print(updated_data)
    
    # Save the updated JSON file
    # with open('updated_' + file_name, 'w') as f:
    #     json.dump(updated_data, f, indent=4)
    
    # print(f"Updated annotations saved to updated_{file_name}")

# Call the function with the file name
update_annotations_with_consensus('full_anotation.json')
