import glob
import io
import os

import pygame
from nubrain.experiment.global_config import GlobalConfig
from PIL import Image

global_config = GlobalConfig()
max_img_storage_dimension = global_config.max_img_storage_dimension


def load_and_scale_images(*, image_directory, screen_width, screen_height):
    """
    Loads all PNG and JPEG images from a directory and scales them to fit the screen (to
    be used for stimulus presentation).
    """
    extensions = ("*.png", "*.jpg", "*.jpeg")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_directory, ext)))

    loaded_images = []
    if not image_files:
        print(f"No images found in directory: {image_directory}")
        return []

    print(f"Found {len(image_files)} images.")

    for filepath in image_files:
        try:
            img = pygame.image.load(filepath)
            img_rect = img.get_rect()

            # Calculate scaling factor to fit screen while maintaining aspect ratio.
            scale_w = screen_width / img_rect.width
            scale_h = screen_height / img_rect.height
            scale = min(scale_w, scale_h)

            new_width = int(img_rect.width * scale)
            new_height = int(img_rect.height * scale)

            scaled_img = pygame.transform.smoothscale(img, (new_width, new_height))
            loaded_images.append({"image_filepath": filepath, "image": scaled_img})
            print(f"Loaded and scaled: {filepath}")
        except pygame.error as e:
            print(f"Error loading or scaling image {filepath}: {e}")
    return loaded_images


def load_image_as_bytes(*, image_path: str):
    """
    Load an image file from disk and return it as a bytes object.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


def resize_image(*, image_bytes: bytes, return_image_file_extension: bool = False):
    """
    Resize image to maximal size (not used for stimulus presentation, but for logging).
    """
    image = Image.open(io.BytesIO(image_bytes))

    if image.format == "JPEG":
        image_format = "JPEG"
        image_file_extension = ".jpg"
    elif image.format == "PNG":
        image_format = "PNG"
        image_file_extension = ".png"
    elif image.format == "WEBP":
        image_format = "WEBP"
        image_file_extension = ".webp"
    else:
        print(f"Unexpected image format, will use png: {image.format}")
        image_format = "PNG"
        image_file_extension = ".png"

    width, height = image.size

    # Check if resizing is needed.
    if (width > max_img_storage_dimension) or (height > max_img_storage_dimension):
        # Calculate the new size maintaining the aspect ratio.
        if width > height:
            new_width = max_img_storage_dimension
            new_height = int(max_img_storage_dimension * height / width)
        else:
            new_height = max_img_storage_dimension
            new_width = int(max_img_storage_dimension * width / height)

        # Resize the image.
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Convert the image to bytes again.
    image_bytes_resized = io.BytesIO()

    image.save(image_bytes_resized, format=image_format)
    image_bytes_resized = bytearray(image_bytes_resized.getvalue())

    if return_image_file_extension:
        return image_bytes_resized, image_file_extension
    else:
        return image_bytes_resized
