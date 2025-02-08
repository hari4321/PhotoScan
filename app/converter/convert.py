import os
import cv2
import numpy as np
from PIL import Image
from app.core.face_detector import detect_faces
from app.core.face_alignment import align_face
from app.core.feature_extraction import extract_face_embedding
from app.data.database import is_image_in_db, insert_reference_image, insert_group_image

# Import HEIF/HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not installed. HEIF/HEIC support won't be available.")

# Import RAW image support
try:
    import rawpy
    import imageio
except ImportError:
    print("rawpy or imageio not installed. RAW image support won't be available.")


def convert_to_jpg(input_path, quality=85, optimize=True, progressive=False):
    """
    Converts an image (RAW, HEIF, or other formats) to JPG format and returns the new path.

    :param input_path: Path to the input image.
    :param quality: Quality of the output JPG (1-100).
    :param optimize: Boolean to optimize the image.
    :param progressive: Boolean to create a progressive JPEG.
    :return: Path to the converted JPG image.
    """
    file_ext = os.path.splitext(input_path)[1].lower()
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}.jpg"

    try:
        if file_ext in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2']:  # RAW file formats
            with rawpy.imread(input_path) as raw:
                rgb_image = raw.postprocess()
                imageio.imsave(output_path, rgb_image, quality=quality)
        else:
            with Image.open(input_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(output_path, "JPEG", quality=quality, optimize=optimize, progressive=progressive)
        
        print(f"Converted: {input_path} -> {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None

def load_image(image_path):
    """
    Loads an image from various formats (RAW, HEIF, etc.) and converts it to an OpenCV-compatible format.
    
    :param image_path: Path to the input image.
    :return: OpenCV (BGR) image or None if failed.
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    
    try:
        if file_ext in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2']:  # RAW file formats
            with rawpy.imread(image_path) as raw:
                rgb_image = raw.postprocess()
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        else:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Convert to RGB
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

