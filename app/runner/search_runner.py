# app/search_runner.py
import numpy as np
from app.data.database import fetch_all_reference_images, fetch_all_group_images
from app.core.feature_extraction import extract_face_embedding
from app.core.face_detector import detect_faces
from app.core.face_alignment import align_face
import cv2

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def process_group_image(image_path, model_name="Facenet", similarity_threshold=0.8):
    print(f"Processing image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return

    # Detect faces
    results = detect_faces(image_path, save_output=False)
    if not results:
        print(f"No face detected in {image_path}.")
        return

    # Align face
    aligned_face = align_face(img, results[0])
    input_embedding = extract_face_embedding(aligned_face, model_name=model_name)

    # Fetch reference images from the database
    ref_images = fetch_all_reference_images()
    if not ref_images:
        print("No reference images found in the database.")
        return

    # Compare with reference images
    for ref_image in ref_images:
        similarity = cosine_similarity(input_embedding, np.array(ref_image['feature_vector']))
        if similarity >= similarity_threshold:
            print(f"Match found with '{ref_image['image_name']}' (Similarity: {similarity:.2f})")
        else:
            print(f"No match for '{ref_image['image_name']}' (Similarity: {similarity:.2f})")
