import glob
import io
import os

import pygame
from PIL import Image


def load_and_scale_images(*, image_directory, screen_width, screen_height):
    """
    Loads all PNG and JPEG images from a directory and scales them to fit the screen.
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

            # Calculate scaling factor to fit screen while maintaining aspect ratio
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


def resize_image(*, image_bytes, return_image_file_extension=False):
    """
    Resize image to maximal size before sending over network.
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

    # Maximum dimension.
    max_dimension = 256

    # Check if resizing is needed.
    if (width > max_dimension) or (height > max_dimension):
        # Calculate the new size maintaining the aspect ratio.
        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension * height / width)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * width / height)

        # Resize the image.
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Convert the image to bytes.
    img_byte_arr = io.BytesIO()

    image.save(img_byte_arr, format=image_format)
    img_byte_arr = bytearray(img_byte_arr.getvalue())

    if return_image_file_extension:
        return img_byte_arr, image_file_extension
    else:
        return img_byte_arr
