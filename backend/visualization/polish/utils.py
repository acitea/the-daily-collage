"""
Image preprocessing utilities for polish providers.

Provides common image conversion and preparation functions that all
providers can use to standardize image inputs before API calls.
"""

import logging
import io
from typing import Tuple, Literal, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def prepare_image(
    image_data: bytes,
    target_width: int = 1344,
    target_height: int = 768,
    output_format: Literal["JPEG", "PNG"] = "JPEG",
    quality: int = 95,
    convert_to_rgb: bool = True,
) -> bytes:
    """
    Prepare and optimize an image for API submission.

    This function handles:
    - Loading image from bytes
    - Resizing to target dimensions
    - Converting color modes (RGBA -> RGB for JPEG)
    - Re-encoding to the specified format

    Args:
        image_data: Input image bytes (PNG, JPEG, etc.)
        target_width: Target width in pixels
        target_height: Target height in pixels
        output_format: Output format ("JPEG" or "PNG")
        quality: JPEG quality (1-100, only used for JPEG)
        convert_to_rgb: Whether to convert RGBA/PA/P to RGB

    Returns:
        bytes: Processed image data in the specified format

    Example:
        >>> image_bytes = open("input.png", "rb").read()
        >>> processed = prepare_image(image_bytes, 1024, 1024, "JPEG")
        >>> open("output.jpg", "wb").write(processed)
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data))
        original_size = image.size
        
        # Resize if needed
        if image.size != (target_width, target_height):
            image = image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            logger.debug(
                f"Resized image from {original_size} to {target_width}x{target_height}"
            )
        
        # Convert color mode if needed
        if convert_to_rgb and image.mode in ("RGBA", "PA", "P"):
            original_mode = image.mode
            image = image.convert("RGB")
            logger.debug(f"Converted image from {original_mode} to RGB")
        
        # Re-encode to target format
        output_bytes = io.BytesIO()
        save_kwargs = {"format": output_format}
        
        if output_format == "JPEG":
            save_kwargs["quality"] = quality
        
        image.save(output_bytes, **save_kwargs)
        result = output_bytes.getvalue()
        
        logger.info(
            f"Prepared image: {len(image_data)} -> {len(result)} bytes "
            f"({output_format}, {target_width}x{target_height})"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to prepare image: {e}")
        # Return original on error
        return image_data


def image_to_base64(
    image_data: bytes,
    mime_type: str = "image/jpeg",
) -> str:
    """
    Convert image bytes to base64 data URI.

    Useful for APIs that require base64-encoded images (e.g., Replicate).

    Args:
        image_data: Image bytes
        mime_type: MIME type for the data URI (e.g., "image/jpeg", "image/png")

    Returns:
        str: Base64 data URI (e.g., "data:image/jpeg;base64,/9j/4AAQ...")

    Example:
        >>> jpeg_bytes = prepare_image(image_data, output_format="JPEG")
        >>> data_uri = image_to_base64(jpeg_bytes, "image/jpeg")
        >>> print(data_uri[:50])
        data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQAB...
    """
    import base64
    
    encoded = base64.b64encode(image_data).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{encoded}"
    
    logger.debug(
        f"Converted {len(image_data)} bytes to base64 data URI "
        f"({len(data_uri)} chars)"
    )
    
    return data_uri


def get_image_dimensions(image_data: bytes) -> Tuple[int, int]:
    """
    Get the dimensions of an image without full processing.

    Args:
        image_data: Image bytes

    Returns:
        Tuple[int, int]: (width, height)

    Example:
        >>> width, height = get_image_dimensions(image_bytes)
        >>> print(f"Image is {width}x{height}")
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        return image.size
    except Exception as e:
        logger.error(f"Failed to get image dimensions: {e}")
        return (0, 0)


def validate_image(
    image_data: bytes,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    allowed_formats: Optional[list[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate image dimensions and format.

    Args:
        image_data: Image bytes to validate
        min_width: Minimum width in pixels
        min_height: Minimum height in pixels
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        allowed_formats: List of allowed formats (e.g., ["JPEG", "PNG"])

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)

    Example:
        >>> valid, error = validate_image(
        ...     image_data,
        ...     min_width=512,
        ...     max_width=2048,
        ...     allowed_formats=["JPEG", "PNG"]
        ... )
        >>> if not valid:
        ...     print(f"Invalid image: {error}")
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        
        # Check dimensions
        if min_width and width < min_width:
            return False, f"Image width {width} < minimum {min_width}"
        
        if min_height and height < min_height:
            return False, f"Image height {height} < minimum {min_height}"
        
        if max_width and width > max_width:
            return False, f"Image width {width} > maximum {max_width}"
        
        if max_height and height > max_height:
            return False, f"Image height {height} > maximum {max_height}"
        
        # Check format
        if allowed_formats and image.format not in allowed_formats:
            return False, f"Image format {image.format} not in {allowed_formats}"
        
        return True, None
        
    except Exception as e:
        return False, f"Failed to validate image: {e}"
