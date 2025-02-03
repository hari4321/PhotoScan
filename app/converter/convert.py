from PIL import Image
import os

# Import and register HEIF/HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not installed. HEIF/HEIC support won't be available.")

def convert_to_jpg(input_path, output_path=None, quality=85, optimize=True, progressive=False):
    """
    Converts an image to JPG format, including HEIF/HEIC support.

    :param input_path: Path to the input image.
    :param output_path: Path for the output JPG image.
    :param quality: Quality of the output JPG (1-100).
    :param optimize: Boolean to optimize the image.
    :param progressive: Boolean to create a progressive JPEG.
    """
    try:
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            if not output_path:
                base, _ = os.path.splitext(input_path)
                output_path = f"{base}.jpg"
            rgb_img.save(output_path, "JPEG", quality=quality, optimize=optimize, progressive=progressive)
            print(f"Converted: {input_path} -> {output_path} (Quality: {quality}, Optimize: {optimize}, Progressive: {progressive})")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
