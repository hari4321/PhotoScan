# app/add_image_data.py
import os
import cv2
from app.core.face_detector import detect_faces
from app.core.face_alignment import align_face
from app.core.feature_extraction import extract_face_embedding
from app.data.database import insert_reference_image, insert_group_image, is_image_in_db

def add_reference_image(image_path, model_name="Facenet"):
    filename = os.path.basename(image_path)
    if is_image_in_db("reference_images", filename):
        print(f"Reference image '{filename}' is already in the database.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return

    results = detect_faces(image_path, save_output=False)
    if not results:
        print(f"No face detected in {image_path}.")
        return

    try:
        aligned_face = align_face(img, results[0])
        embedding = extract_face_embedding(aligned_face, model_name=model_name)
        insert_reference_image(filename, embedding)
        print(f"Reference image '{filename}' added successfully.")
    except Exception as e:
        print(f"Error processing '{filename}': {e}")

def add_group_image(image_path):
    filename = os.path.basename(image_path)
    if is_image_in_db("group_images", filename):
        print(f"Group image '{filename}' is already in the database.")
        return

    insert_group_image(filename)
    print(f"Group image '{filename}' added successfully.")

def add_group_images_from_folder(folder_path):
    image_extensions = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    if not images:
        print(f"No images found in folder '{folder_path}'.")
        return

    print(f"Adding {len(images)} images from folder '{folder_path}'...")

    for image_file in images:
        image_path = os.path.join(folder_path, image_file)
        add_group_image(image_path)

    print(f"Finished adding images from '{folder_path}'.")
