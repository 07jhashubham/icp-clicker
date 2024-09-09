import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Function to visualize the image and bounding boxes
def visualize_image(json_file, image_index, image_folder):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Find the image details based on the provided index
    image_info = None
    for img in data['images']:
        if img['id'] == image_index:
            image_info = img
            break

    if not image_info:
        print(f"No image found with index {image_index}")
        return

    # Load the image file
    image_path = os.path.join(image_folder, image_info['file_name'])
    image = Image.open(image_path)

    # Get annotations for the image
    annotations = [anno for anno in data['annotations'] if anno['image_id'] == image_index]

    # Set up the plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot each original bounding box
    for annotation in annotations:
        for i in range(1, 5):
            bbox_key = f'bbox{i}'
            if bbox_key in annotation:
                bbox = annotation[bbox_key]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none', label=f'User {i}'
                )
                ax.add_patch(rect)

        # Plot the final consensus bounding box
        if 'final_bbox' in annotation:
            final_bbox = annotation['final_bbox']
            rect = patches.Rectangle(
                (final_bbox[0], final_bbox[1]), final_bbox[2], final_bbox[3],
                linewidth=2, edgecolor='g', facecolor='none', label='Consensus'
            )
            ax.add_patch(rect)

    # Set plot title
    plt.title(f"Image ID: {image_index} - {image_info['file_name']}")
    plt.legend(handles=[patches.Patch(color='red', label='User Annotations'),
                        patches.Patch(color='green', label='Consensus')])
    
    # Show the plot
    plt.show()

# Main function to get input and visualize
if __name__ == "__main__":
    # Take image index as input from the user
    image_index = int(input("Enter the image index to visualize: "))
    
    # Define the file paths
    json_file = 'updated_modified_annotations_with_outliers.json'
    image_folder = 'test'

    # Call the visualization function
    visualize_image(json_file, image_index, image_folder)
