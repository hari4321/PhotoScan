# main.py
import os
from dotenv import load_dotenv
from app.data.database import init_db, is_image_in_db
from app.runner.add_image_runner import add_reference_image, add_group_image, add_group_images_from_folder
from app.runner.search_runner import process_group_image

# Load environment variables
load_dotenv()

# Initialize the database
init_db()

# Read environment variables
MODE = os.getenv("MODE")
IMAGE_PATH = os.getenv("IMAGE_PATH")
FOLDER_PATH = os.getenv("FOLDER_PATH")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.8))
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet")
ADD_MODE = os.getenv("ADD_MODE", "single")
IMAGE_TYPE = os.getenv("IMAGE_TYPE", "ref")

def main():
    if MODE == "add":
        if IMAGE_TYPE == "ref":
            if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
                print(f"Error: IMAGE_PATH '{IMAGE_PATH}' is invalid or does not exist.")
                return
            add_reference_image(IMAGE_PATH, model_name=MODEL_NAME)

        elif IMAGE_TYPE == "group":
            if ADD_MODE == "folder":
                if not FOLDER_PATH or not os.path.isdir(FOLDER_PATH):
                    print(f"Error: FOLDER_PATH '{FOLDER_PATH}' is invalid or does not exist.")
                    return
                add_group_images_from_folder(FOLDER_PATH)
            elif ADD_MODE == "single":
                if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
                    print(f"Error: IMAGE_PATH '{IMAGE_PATH}' is invalid or does not exist.")
                    return
                add_group_image(IMAGE_PATH)
            else:
                print(f"Error: Invalid ADD_MODE '{ADD_MODE}'. Use 'single' or 'folder'.")
        else:
            print(f"Error: Invalid IMAGE_TYPE '{IMAGE_TYPE}'. Use 'ref' or 'group'.")

    elif MODE == "search":
        if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
            print(f"Error: IMAGE_PATH '{IMAGE_PATH}' is invalid or does not exist.")
            return

        filename = os.path.basename(IMAGE_PATH)
        
        # Check if reference image is already in the database
        if not is_image_in_db("reference_images", filename):
            print(f"Reference image '{filename}' not found in the database. Adding it now...")
            add_reference_image(IMAGE_PATH, model_name=MODEL_NAME)
        
        # Proceed with the search
        process_group_image(IMAGE_PATH, model_name=MODEL_NAME, similarity_threshold=SIMILARITY_THRESHOLD)

    else:
        print(f"Error: Invalid MODE '{MODE}'. Use 'add' or 'search'.")

if __name__ == "__main__":
    main()
