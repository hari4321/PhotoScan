import os
from dotenv import load_dotenv
from app.converter.convert import convert_to_jpg

def main():
    load_dotenv()  # Load variables from .env file

    input_path = os.getenv('INPUT_PATH')
    output_path = os.getenv('OUTPUT_PATH')
    quality = int(os.getenv('QUALITY', 85))
    optimize = os.getenv('OPTIMIZE', 'True').lower() in ['true', '1', 'yes']
    progressive = os.getenv('PROGRESSIVE', 'False').lower() in ['true', '1', 'yes']

    if not input_path:
        print("Error: INPUT_PATH not specified in .env file.")
        return

    convert_to_jpg(input_path, output_path, quality, optimize, progressive)

if __name__ == "__main__":
    main()
