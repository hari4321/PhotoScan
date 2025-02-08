from app.data.database import get_reference_embedding, get_all_group_embeddings 
from app.core.similarity_matching import match_embeddings

def search_in_group_images(image_path, metric="cosine", similarity_threshold=0.8):
    print("Processing group images from the database...")

    # Fetch group images and their embeddings from the database
    group_embeddings = get_all_group_embeddings()
    if not group_embeddings:
        print("No group images found in the database.")
        return

    # Fetch reference embedding from the database
    reference_embedding = get_reference_embedding(image_path)
    if reference_embedding is None:
        print(f"No reference embedding found for {image_path}.")
        return

    # Compare all group images with the reference embedding
    matches = match_embeddings(group_embeddings, reference_embedding, metric=metric, threshold=similarity_threshold)

    if matches:
        for group_filename, score in matches.items():
            print(f"Match found: {group_filename} ↔ {image_path} (Score: {score:.2f})")
    else:
        print(f"No matches found for {image_path}.")
        