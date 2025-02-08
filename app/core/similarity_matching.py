import numpy as np
from scipy.spatial.distance import cosine, euclidean

def compute_cosine_similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two embeddings.
    
    Args:
        embedding1 (np.array): First embedding vector.
        embedding2 (np.array): Second embedding vector.
    
    Returns:
        float: Cosine similarity value between -1 and 1.
    """
    # The cosine function from scipy returns the cosine distance.
    # To convert it to similarity, we can subtract from 1.
    return 1 - cosine(embedding1, embedding2)

def compute_euclidean_distance(embedding1, embedding2):
    """
    Computes the Euclidean distance between two embeddings.
    
    Args:
        embedding1 (np.array): First embedding vector.
        embedding2 (np.array): Second embedding vector.
    
    Returns:
        float: Euclidean distance.
    """
    return euclidean(embedding1, embedding2)

def match_embeddings(group_embeddings, reference_embedding, metric="cosine", threshold=None):
    """
    Matches a set of group embeddings against a single reference embedding.

    Args:
        group_embeddings (dict): A dictionary where keys are identifiers (e.g., filenames)
                                 and values are the embedding vectors.
        reference_embedding (np.array): The embedding vector to compare against.
        metric (str): The similarity metric to use ("cosine" or "euclidean").
        threshold (float): A threshold to determine a match.
                           - For cosine, higher values indicate similarity (e.g., 0.8).
                           - For Euclidean, lower values indicate similarity.

    Returns:
        dict: A dictionary with keys as group image identifiers and values as the computed similarity
              (or distance) scores for those that pass the threshold.
    """
    matches = {}

    for group_id, group_embedding in group_embeddings.items():
        if metric == "cosine":
            similarity = compute_cosine_similarity(group_embedding, reference_embedding)
            if threshold is None or similarity >= threshold:
                matches[group_id] = similarity
        elif metric == "euclidean":
            distance = compute_euclidean_distance(group_embedding, reference_embedding)
            if threshold is None or distance <= threshold:
                matches[group_id] = distance
        else:
            raise ValueError("Unsupported metric. Choose either 'cosine' or 'euclidean'.")

    return matches
