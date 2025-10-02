# MIT License
#
# Copyright (c) 2025 Groq, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Source: https://github.com/groq/openbench
# Modifications: Compl-AI Team


def detect_image_mime_type(image_bytes: bytes) -> str:
    """
    Detect the MIME type of an image from its bytes.

    Uses magic bytes to detect common image formats.
    Falls back to 'image/png' if detection fails.

    Args:
        image_bytes: Raw image bytes

    Returns:
        MIME type string (e.g., 'image/png', 'image/jpeg', 'image/webp')
    """
    try:
        # Use magic bytes to detect image format
        return _detect_from_magic_bytes(image_bytes)

    except Exception:
        # Fallback to PNG if detection fails
        return "image/png"


def _detect_from_magic_bytes(image_bytes: bytes) -> str:
    """
    Detect image format from magic bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        MIME type string
    """
    if len(image_bytes) < 4:
        return "image/png"

    # Check for common image format signatures
    signatures = [
        (b"\xff\xd8\xff", "image/jpeg"),  # JPEG
        (b"\x89PNG\r\n\x1a\n", "image/png"),  # PNG
        (b"GIF87a", "image/gif"),  # GIF87a
        (b"GIF89a", "image/gif"),  # GIF89a
        (b"BM", "image/bmp"),  # BMP
        (b"RIFF", "image/webp"),  # WebP (RIFF header)
        (b"II*\x00", "image/tiff"),  # TIFF little-endian
        (b"MM\x00*", "image/tiff"),  # TIFF big-endian
        (b"\x00\x00\x01\x00", "image/ico"),  # ICO
        (b"\x00\x00\x02\x00", "image/ico"),  # ICO
    ]

    for signature, mime_type in signatures:
        if image_bytes.startswith(signature):
            return mime_type

    # Default fallback
    return "image/png"
