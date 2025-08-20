import glob
import io
import os

import pygame
from PIL import Image

from nubrain.experiment.global_config import GlobalConfig

global_config = GlobalConfig()
max_img_storage_dimension = global_config.max_img_storage_dimension


def get_all_image_paths(*, image_directory: str):
    """
    Get the path of all images in target directory.
    """
    extensions = ("*.png", "*.jpg", "*.jpeg")
    image_file_paths = []
    for ext in extensions:
        image_file_paths.extend(glob.glob(os.path.join(image_directory, ext)))

    if not image_file_paths:
        print(f"No images found in directory: {image_directory}")
        return []

    print(f"Found {len(image_file_paths)} images.")

    return image_file_paths


def load_and_scale_image(
    *,
    image_file_path: str,
    screen_width: int,
    screen_height: int,
):
    """
    Loads image as pygame object and scale them to fit the screen, if the image is
    larger than the screen (to be used for stimulus presentation).
    """

    # (1) Load image.
    try:
        image = pygame.image.load(image_file_path)
        img_rect = image.get_rect()

        if ((screen_width * 0.5) < img_rect.width) or (
            (screen_height * 0.5) < img_rect.height
        ):
            # The image is larger than the screen. Scale it to fit the screen while
            # maintaining the aspect ratio.
            scale_w = screen_width * 0.5 / img_rect.width
            scale_h = screen_height * 0.5 / img_rect.height
            scale = min(scale_w, scale_h)

            new_width = int(img_rect.width * scale)
            new_height = int(img_rect.height * scale)
            image = pygame.transform.smoothscale(image, (new_width, new_height))

        else:
            # The image is not too large for the screen.
            pass

    except pygame.error as e:
        print(f"Error loading or scaling image {image_file_path}: {e}")
        return None

    # (2) Load image category from text file (if it exists).
    try:
        file_path_withou_extension = os.path.splitext(image_file_path)[0]
        path_txt = file_path_withou_extension + ".txt"

        if os.path.isfile(path_txt):
            with open(path_txt, "r", encoding="utf-8") as file:
                image_category = file.read()
            image_category = image_category.strip()
        else:
            image_category = "null"  # Needs to be string for saving to hdf5
    except Exception as e:
        print(f"Error loading image metadata {path_txt}: {e}")
        return None

    image_and_metadata = {
        "image_file_path": image_file_path,
        "image": image,
        "image_category": image_category,
    }

    return image_and_metadata


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
