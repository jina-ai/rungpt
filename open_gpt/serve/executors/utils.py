import io

from PIL import Image


def convert_image_to_rgb(image):
    return image.convert('RGB')


def blob2image(blob: bytes):
    return Image.open(io.BytesIO(blob))
