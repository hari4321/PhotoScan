from deepface import DeepFace
import numpy as np

def extract_face_embedding(face_img, model_name="Facenet"):
    """
    Extracts the embedding of an aligned face image using the specified model.
    
    Args:
        face_img (numpy.ndarray): The aligned face image in RGB format.
        model_name (str): Name of the model to use. Options include "Facenet", "ArcFace", "VGG-Face", etc.
        
    Returns:
        numpy.ndarray: The embedding vector for the face.
    """
    # DeepFace expects a BGR image by default, but you can pass RGB if enforce_detection is off
    # We use enforce_detection=False since the face is already detected and aligned
    try:
        # The `represent` function returns a list of dictionaries; we take the first result.
        representation = DeepFace.represent(face_img, model_name=model_name, enforce_detection=False)
        embedding = representation[0]["embedding"]
        # Ensure the embedding is a numpy array
        return np.array(embedding)
    except Exception as e:
        raise RuntimeError(f"Error extracting embedding: {e}")
