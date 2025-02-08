import sqlite3
import json
import os
import numpy as np

# Define the database path (it will be created inside the app folder)
DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

def init_db():
    data_dir = os.path.dirname(DB_PATH)
    os.makedirs(data_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reference_images (
                        id INTEGER PRIMARY KEY,
                        filename TEXT UNIQUE,
                        embedding BLOB
                      )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS group_images (
                        id INTEGER PRIMARY KEY,
                        filename TEXT UNIQUE,
                        embedding BLOB
                      )''')
    conn.commit()
    conn.close()

def insert_reference_image(filename, embedding):
    """
    Inserts or updates a reference image embedding into the database.
    
    Args:
        filename (str): The filename (or identifier) of the reference image.
        embedding (list or np.array): The face embedding vector.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    embedding_json = json.dumps(embedding)
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO reference_images (filename, embedding)
            VALUES (?, ?)
        ''', (filename, embedding_json))
        conn.commit()
    except Exception as e:
        print(f"Error inserting reference image {filename}: {e}")
    finally:
        conn.close()

def get_all_reference_embeddings():
    """
    Retrieves all reference embeddings from the database.
    
    Returns:
        dict: A dictionary where keys are filenames and values are the embedding vectors.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, embedding FROM reference_images')
    rows = cursor.fetchall()
    reference_embeddings = {}
    for filename, embedding_json in rows:
        try:
            embedding = json.loads(embedding_json)
            reference_embeddings[filename] = embedding
        except Exception as e:
            print(f"Error parsing embedding for {filename}: {e}")
    conn.close()
    return reference_embeddings

def get_all_group_embeddings():
    """
    Retrieves all group image embeddings from the database.
    
    Returns:
        dict: A dictionary where keys are filenames and values are embedding vectors.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, embedding FROM group_images")
    rows = cursor.fetchall()
    conn.close()

    group_embeddings = {}
    for filename, embedding_json in rows:
        try:
            embedding = np.array(json.loads(embedding_json))  # Convert to NumPy array
            group_embeddings[filename] = embedding
        except Exception as e:
            print(f"Error parsing embedding for {filename}: {e}")

    return group_embeddings

def insert_group_image(filename, embedding):
    """
    Inserts a group image record into the database.
    
    Args:
        filename (str): The filename (or identifier) of the group image.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    embedding_json = json.dumps(embedding)
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO group_images (filename, embedding)
            VALUES (?, ?)
        ''', (filename, embedding_json))
        conn.commit()
    except Exception as e:
        print(f"Error inserting reference image {filename}: {e}")
    finally:
        conn.close()

def is_image_in_db(table, filename):
    """
    Checks if an image exists in the specified table.
    
    Args:
        table (str): The table name ('reference_images' or 'group_images').
        filename (str): The filename to check.
    
    Returns:
        bool: True if the image exists, False otherwise.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE filename = ?", (filename,))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists

def get_reference_embedding(image_path):
    """
    Retrieves the embedding of a specific reference image from the database.

    Args:
        image_path (str): The file path of the image.

    Returns:
        list or None: The embedding vector if found, otherwise None.
    """
    image_filename = os.path.basename(image_path)  # Extract only the filename
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT embedding FROM reference_images WHERE filename = ?', (image_filename,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        try:
            return json.loads(row[0])  # Convert JSON string back to a list
        except Exception as e:
            print(f"Error parsing embedding for {image_filename}: {e}")
            return None
    else:
        print(f"Embedding not found for {image_filename}.")
        return None