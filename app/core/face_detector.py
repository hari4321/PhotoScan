import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os

class ImagePathError(Exception):
    """Custom exception raised when no image path is provided."""
    def __init__(self, message="No image path provided. Please specify a valid image file path."):
        self.message = message
        super().__init__(self.message)

def detect_faces(image_path, save_output=False, output_filename="detected_image.jpg"):
    """
    Detect faces in the given image and optionally save the output image with bounding boxes.
    
    Args:
        image_path (str): Path to the input image.
        save_output (bool): If True, save the output image with detected face boxes.
        output_filename (str): The filename to use when saving the output image.
        
    Returns:
        list: A list of dictionaries, each containing the detected face's details.
    
    Raises:
        ImagePathError: If image_path is not provided.
        ValueError: If the image cannot be read from the provided path.
    """
    # Check if an image path is provided
    if not image_path:
        raise ImagePathError()

    # Initialize the MTCNN face detector
    detector = MTCNN()
    
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    
    # Convert image from BGR (OpenCV default) to RGB for processing and visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform face detection
    results = detector.detect_faces(img_rgb)
    
    # Debug: Print detected face details
    for i, face in enumerate(results):
        print(f"Face {i+1}: {face}")
        # Each face dict contains:
        # - 'box': [x, y, width, height]
        # - 'confidence': detection confidence score
        # - 'keypoints': facial landmarks (e.g., left_eye, right_eye, nose, mouth_left, mouth_right)
    
    # Draw bounding boxes on the image for each detected face
    for face in results:
        x, y, width, height = face['box']
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # If save_output is True, save the image inside the current directory
    if save_output:
        output_dir = os.path.dirname(__file__)  # Ensure correct directory
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Use plt.imsave to save an RGB image
        plt.imsave(output_path, img_rgb)
        print(f"Output saved to: {output_path}")
    # else:
    #     # Otherwise, display the image with detections
    #     plt.imshow(img_rgb)
    #     plt.title('Detected Faces')
    #     plt.axis('off')
    #     plt.show()
    
    return results
