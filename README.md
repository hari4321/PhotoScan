# PhotoScan

**PhotoScan** is an advanced facial recognition system designed to identify and match faces across groups of images with a reference dataset. It leverages cutting-edge machine learning techniques for face detection, alignment, feature extraction, and similarity matching to ensure high accuracy. The project is modular, allowing easy expansion or fine-tuning for diverse applications such as security systems, photo organization, and identity verification.

Key features include:

- **Face Detection**: Using MTCNN to accurately detect faces in group images and reference photos.
- **Face Alignment**: Ensuring facial features are properly aligned for consistent feature extraction.
- **Feature Extraction**: Leveraging FaceNet for creating unique facial embeddings.
- **Similarity Matching**: Using cosine similarity to match group images with reference images stored in a structured SQLite database.
- **Efficient Database Management**: Organized storage of embeddings and metadata, with easy updating and retrieval.
- **Modular Execution**: Flexible execution modes through environment variable settings, supporting batch processing or single image updates.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository
```
git clone https://github.com/hari4321/PhotoScan.git
cd PhotoScan
```
### 2. Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Virtual Environment
```
# MODE options: add, search
MODE=search

# Image path for single image operations
IMAGE_PATH= # Add path here

# Folder path for adding multiple group images
FOLDER_PATH= # Add path here

# IMAGE_TYPE options: ref, group
IMAGE_TYPE=group

# Add mode options: single, folder (only applies when MODE=add and IMAGE_TYPE=group)
ADD_MODE=folder                      

# Settings for search
SIMILARITY_THRESHOLD=0.8
MODEL_NAME=Facenet
```

### 4. Usage
```
python main.py
```
