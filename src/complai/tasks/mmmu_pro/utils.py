import base64
import io
import math
import re
from typing import Any

from PIL import Image


# Strictest provider limit on the base64 image payload: Anthropic allows 10 MB, but
# Bedrock and Vertex (reachable through OpenRouter fallbacks) allow 5 MB.
MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024


def _base64_size(byte_count: int) -> int:
    return (byte_count + 2) // 3 * 4


def shrink_image(
    image_bytes: bytes, max_base64_bytes: int = MAX_IMAGE_BASE64_BYTES
) -> bytes:
    """Downscale a PNG until its base64 encoding fits ``max_base64_bytes``.

    Images that already fit are returned unchanged, so only a handful of very
    large photos are touched. Sample hashes do not cover image bytes, so this
    causes no drift.
    """
    if _base64_size(len(image_bytes)) <= max_base64_bytes:
        return image_bytes
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    data, scale = image_bytes, 1.0
    for _ in range(10):
        scale *= 0.95 * math.sqrt(max_base64_bytes / _base64_size(len(data)))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        buffer = io.BytesIO()
        image.resize(size, Image.Resampling.LANCZOS).save(buffer, format="PNG")
        data = buffer.getvalue()
        if _base64_size(len(data)) <= max_base64_bytes:
            return data
    raise ValueError(f"Could not shrink image below {max_base64_bytes} base64 bytes")


def image_to_data_uri(image_data: dict[str, Any]) -> str:
    """Convert image data from HuggingFace to a base64 data URI.

    Images over ``MAX_IMAGE_BASE64_BYTES`` are downscaled first (see ``shrink_image``).

    Args:
        image_data: Image data as dictionary with 'bytes' key.

    Returns:
        Base64 data URI string for the image.
    """
    if isinstance(image_data, dict) and "bytes" in image_data:
        image_bytes = image_data["bytes"]
    else:
        raise ValueError("Image data expected to be a dictionary with 'bytes' key")

    b64_data = base64.b64encode(shrink_image(image_bytes)).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


def replace_image_tokens(record: dict[str, Any]) -> tuple[str, list[int]]:
    """Replace image placeholders with generic <image> token and extract order.

    This matches the official MMMU-Pro implementation.

    Args:
        input_string: Text containing placeholders like <image 1>, <image 2>.

    Returns:
        Tuple of (modified text with <image> placeholders, list of image indices in order).
    """
    question_text = record["question"]
    modified_text = re.sub(r"<image\s+\d+>", "<image>", question_text)

    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", question_text)]
    if not image_order:
        # No placeholders found - add all available images
        # 51 samples have no text placeholders but images as options
        for i in range(1, 8):
            image = record.get(f"image_{i}")
            if image is not None:
                image_order.append(i)

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
