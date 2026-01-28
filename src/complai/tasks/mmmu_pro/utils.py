import base64
import re
from typing import Any


def image_to_data_uri(image_data: dict[str, Any]) -> str:
    """Convert image data from HuggingFace to a base64 data URI.

    Args:
        image_data: Image data as dictionary with 'bytes' key.

    Returns:
        Base64 data URI string for the image.
    """
    if isinstance(image_data, dict) and "bytes" in image_data:
        image_bytes = image_data["bytes"]
    else:
        raise ValueError("Image data expected to be a dictionary with 'bytes' key")

    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


def replace_image_tokens(input_string: str) -> tuple[str, list[int]]:
    """Replace image placeholders with generic <image> token and extract order.

    This matches the official MMMU-Pro implementation.

    Args:
        input_string: Text containing placeholders like <image 1>, <image 2>.

    Returns:
        Tuple of (modified text with <image> placeholders, list of image indices in order).
    """
    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", input_string)]
    modified_text = re.sub(r"<image\s+\d+>", "<image>", input_string)

    return modified_text, image_order


def get_images_in_order(
    record: dict[str, Any], image_order: list[int]
) -> list[dict[str, Any]]:
    """Get images from record in the specified order.

    Args:
        record: The dataset record.
        image_order: List of image indices (1-indexed) in order of appearance.

    Returns:
        List of images in the order they appear in the text.
    """
    images = []
    for idx in image_order:
        image_key = f"image_{idx}"
        if image_key in record and record[image_key] is not None:
            images.append(record[image_key])
    return images
