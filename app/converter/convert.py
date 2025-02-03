from PIL import Image
import os

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


def convert_to_jpg(input_path, output_path=None, quality=85, optimize=True, progressive=False):
    """
    Converts an image (RAW, HEIF, or other formats) to JPG format.

    :param input_path: Path to the input image.
    :param output_path: Path for the output JPG image.
    :param quality: Quality of the output JPG (1-100).
    :param optimize: Boolean to optimize the image.
    :param progressive: Boolean to create a progressive JPEG.
    """
    file_ext = os.path.splitext(input_path)[1].lower()

    try:
        if file_ext in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2']:  # RAW file formats
            convert_raw_to_jpg(input_path, output_path, quality, optimize, progressive)
        else:
            with Image.open(input_path) as img:
                rgb_img = img.convert("RGB")
                if not output_path:
                    base, _ = os.path.splitext(input_path)
                    output_path = f"{base}.jpg"
                rgb_img.save(output_path, "JPEG", quality=quality, optimize=optimize, progressive=progressive)
                print(f"Converted: {input_path} -> {output_path} (Quality: {quality}, Optimize: {optimize}, Progressive: {progressive})")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")


def convert_raw_to_jpg(input_path, output_path, quality=85, optimize=True, progressive=False):
    """
    Converts RAW image files to JPG format.

    :param input_path: Path to the RAW image.
    :param output_path: Path for the output JPG image.
    :param quality: Quality of the output JPG (1-100).
    """
    try:
        with rawpy.imread(input_path) as raw:
            rgb_image = raw.postprocess()
            if not output_path:
                base, _ = os.path.splitext(input_path)
                output_path = f"{base}.jpg"
            imageio.imsave(output_path, rgb_image, quality=quality)
            print(f"Converted RAW: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting RAW file {input_path}: {e}")
