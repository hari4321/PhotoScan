import cv2
import numpy as np

def align_face(img, face):
    """
    Aligns the face in the image using the detected keypoints.

    Args:
        img (numpy.ndarray): The original image in RGB.
        face (dict): A dictionary with face detection details including 'box' and 'keypoints'.
            Expected keypoints: 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'.

    Returns:
        aligned_face (numpy.ndarray): The aligned face image cropped around the face.
    """

    # Get the keypoints for the left and right eyes
    keypoints = face.get('keypoints', {})
    if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
        raise ValueError("Face detection result does not contain the required eye keypoints.")

    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Calculate the center of the two eyes and cast to int
    eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
    eye_center_y = int((left_eye[1] + right_eye[1]) / 2)
    eye_center = (eye_center_x, eye_center_y)

    # Calculate the angle between the eyes in degrees
    dy = float(right_eye[1] - left_eye[1])
    dx = float(right_eye[0] - left_eye[0])
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Optional: Define a desired distance between the eyes (in pixels)
    desired_eye_distance = 60.0  # Adjust as needed
    current_eye_distance = np.sqrt(dx**2 + dy**2)
    scale = desired_eye_distance / current_eye_distance

    # Compute the rotation matrix for the affine transformation
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    
    # Get the dimensions of the input image
    (h, w) = img.shape[:2]
    
    # Perform the affine transformation (rotation + scaling)
    aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # Crop the face region based on the face bounding box.
    # Note: The bounding box might not be perfect after rotation, consider refining this if needed.
    x, y, width, height = face['box']
    # Ensure the bounding box values are integers
    x, y, width, height = int(x), int(y), int(width), int(height)
    cropped_face = aligned_img[y:y+height, x:x+width]
    
    return cropped_face
