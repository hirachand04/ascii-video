"""
ASCII Converter Module
======================
This module provides functions to convert video frames to ASCII art.

How ASCII Conversion Works:
---------------------------
1. Convert the frame to grayscale (single brightness channel 0-255)
2. Resize the frame to a smaller resolution (ASCII characters are taller than wide,
   so we compensate by using a ~2:1 width:height ratio)
3. Map each pixel's brightness value to an ASCII character:
   - Dark pixels (low values) → Dense characters like '@', '#', '%'
   - Bright pixels (high values) → Sparse characters like '.', ' '
4. Build a string representation of the frame line by line
"""

import numpy as np
import cv2
from typing import Tuple, Optional


# ASCII character sets ordered from darkest to lightest
# Dense characters represent dark areas, sparse characters represent light areas

# Standard set - 10 levels, good for basic visibility
ASCII_CHARS_DENSE_TO_LIGHT = "@%#*+=-:. "
ASCII_CHARS_LIGHT_TO_DENSE = " .:-=+*#%@"

# Extended character set for more detail - 70 characters for fine gradation
ASCII_CHARS_EXTENDED = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# HIGH CONTRAST set - optimized for object recognition
# Uses characters with very distinct visual weights and shapes
# 16 carefully selected characters for maximum clarity
ASCII_CHARS_HIGH_CONTRAST = "█▓▒░■□●○◆◇★☆+:. "

# Block characters - uses Unicode block elements for smooth gradients
ASCII_CHARS_BLOCKS = "█▓▒░ "

# Simple high-visibility ASCII (no Unicode required)
ASCII_CHARS_SIMPLE_HD = "@#W$9876543210?!abc;:+=-,._ "


def frame_to_ascii(
    frame: np.ndarray,
    width: int = 120,
    charset: str = "standard",
    invert: bool = False
) -> str:
    """
    Convert a video frame (BGR or grayscale) to ASCII art.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame from OpenCV (BGR format or grayscale)
    width : int
        Desired ASCII output width in characters (default: 120)
    charset : str
        Character set to use: "standard", "extended", or custom string
    invert : bool
        If True, invert brightness mapping (light chars for dark pixels)
    
    Returns:
    --------
    str
        ASCII representation of the frame
    """
    if frame is None or frame.size == 0:
        return ""
    
    # Select character set
    if charset == "standard":
        ascii_chars = ASCII_CHARS_LIGHT_TO_DENSE if invert else ASCII_CHARS_DENSE_TO_LIGHT
    elif charset == "extended":
        ascii_chars = ASCII_CHARS_EXTENDED[::-1] if invert else ASCII_CHARS_EXTENDED
    else:
        ascii_chars = charset[::-1] if invert else charset
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate new dimensions maintaining aspect ratio
    # ASCII characters are typically ~2x taller than wide, so we compensate
    height, original_width = gray.shape
    aspect_ratio = height / original_width
    new_height = int(width * aspect_ratio * 0.5)  # 0.5 compensates for char aspect ratio
    
    # Ensure minimum dimensions
    new_height = max(1, new_height)
    width = max(1, width)
    
    # Resize frame
    resized = cv2.resize(gray, (width, new_height), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to index into ASCII character set
    num_chars = len(ascii_chars)
    
    # Map pixel values (0-255) to character indices (0 to num_chars-1)
    # Using vectorized operations for performance
    indices = (resized * (num_chars - 1) / 255).astype(np.uint8)
    
    # Build ASCII string
    ascii_lines = []
    for row in indices:
        line = "".join(ascii_chars[idx] for idx in row)
        ascii_lines.append(line)
    
    return "\n".join(ascii_lines)


def frame_to_ascii_array(
    frame: np.ndarray,
    width: int = 120,
    charset: str = "standard",
    invert: bool = False
) -> list:
    """
    Convert a video frame to ASCII art as a list of strings (one per line).
    
    This is useful when you need to process lines individually.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame from OpenCV
    width : int
        Desired ASCII output width in characters
    charset : str
        Character set to use
    invert : bool
        If True, invert brightness mapping
    
    Returns:
    --------
    list
        List of strings, each representing one line of ASCII art
    """
    ascii_str = frame_to_ascii(frame, width, charset, invert)
    return ascii_str.split("\n") if ascii_str else []


def get_optimal_dimensions(
    frame_shape: Tuple[int, int],
    max_width: int = 120,
    max_height: int = 40
) -> Tuple[int, int]:
    """
    Calculate optimal ASCII dimensions based on frame and constraints.
    
    Parameters:
    -----------
    frame_shape : Tuple[int, int]
        Original frame (height, width)
    max_width : int
        Maximum ASCII width in characters
    max_height : int
        Maximum ASCII height in lines
    
    Returns:
    --------
    Tuple[int, int]
        Optimal (width, height) for ASCII output
    """
    height, width = frame_shape[:2]
    aspect_ratio = height / width
    
    # Start with max width and calculate corresponding height
    ascii_width = max_width
    ascii_height = int(ascii_width * aspect_ratio * 0.5)
    
    # If height exceeds max, scale down
    if ascii_height > max_height:
        ascii_height = max_height
        ascii_width = int(ascii_height / (aspect_ratio * 0.5))
    
    return (max(1, ascii_width), max(1, ascii_height))


def ascii_to_image(
    ascii_art: str,
    font_size: int = 16,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    fg_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Render ASCII art to an image using brightness blocks.
    
    Parameters:
    -----------
    ascii_art : str
        ASCII art string with newlines
    font_size : int
        Block size multiplier
    bg_color : Tuple[int, int, int]
        Background color in BGR
    fg_color : Tuple[int, int, int]
        Foreground (text) color in BGR
    
    Returns:
    --------
    np.ndarray
        Image with rendered ASCII art as brightness blocks
    """
    lines = ascii_art.split("\n")
    if not lines or not lines[0]:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Character set for brightness mapping
    ascii_chars = ASCII_CHARS_DENSE_TO_LIGHT  # "@%#*+=-:. "
    num_chars = len(ascii_chars)
    
    # Block size for each character
    block_size = max(6, font_size // 2)
    
    # Calculate image dimensions
    max_line_length = max(len(line) for line in lines) if lines else 1
    img_width = max_line_length * block_size
    img_height = len(lines) * block_size
    
    # Create black image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Draw blocks based on character brightness
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            # Determine brightness from character
            # '@' = darkest (index 0), ' ' = brightest (index 9)
            try:
                char_idx = ascii_chars.index(char)
            except ValueError:
                char_idx = num_chars // 2  # Default to middle brightness
            
            # Map character index to brightness (0-255)
            # Higher index = lighter character = brighter output
            brightness = int((char_idx / max(1, num_chars - 1)) * 255)
            
            # Calculate block position
            y1 = row_idx * block_size
            y2 = y1 + block_size
            x1 = col_idx * block_size
            x2 = x1 + block_size
            
            # Apply brightness to the foreground color
            color = (
                int(fg_color[0] * brightness / 255),
                int(fg_color[1] * brightness / 255),
                int(fg_color[2] * brightness / 255)
            )
            
            # Draw filled rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    return img


def frame_to_brightness_image(
    frame: np.ndarray,
    width: int = 80,
    block_size: int = 8,
    fg_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Convert frame directly to brightness blocks (bypasses ASCII text).
    This is a simpler, more reliable approach.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input BGR frame
    width : int
        Number of columns (blocks)
    block_size : int
        Size of each block in pixels
    fg_color : Tuple[int, int, int]
        Foreground color in BGR
    
    Returns:
    --------
    np.ndarray
        Image with brightness blocks
    """
    if frame is None or frame.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Calculate dimensions
    height, orig_width = gray.shape
    aspect_ratio = height / orig_width
    new_height = int(width * aspect_ratio * 0.5)  # Compensate for block aspect
    
    # Resize to target ASCII dimensions
    small = cv2.resize(gray, (width, max(1, new_height)), interpolation=cv2.INTER_AREA)
    
    # Create output image
    img_width = width * block_size
    img_height = new_height * block_size
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Draw blocks
    for row in range(new_height):
        for col in range(width):
            brightness = int(small[row, col])
            
            y1 = row * block_size
            y2 = y1 + block_size
            x1 = col * block_size
            x2 = x1 + block_size
            
            color = (
                int(fg_color[0] * brightness / 255),
                int(fg_color[1] * brightness / 255),
                int(fg_color[2] * brightness / 255)
            )
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    return img


def frame_to_ascii_image(
    frame: np.ndarray,
    width: int = 150,
    fg_color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    invert: bool = False,
    mode: str = "standard",
    use_color: bool = False,
    output_width: int = 1280,
    output_height: int = 720
) -> np.ndarray:
    """
    Convert frame to an image with ASCII characters rendered.
    Renders directly at the output resolution for maximum clarity.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input BGR frame
    width : int
        Number of ASCII columns (character columns)
    fg_color : Tuple[int, int, int]
        Text color in BGR (used when use_color=False)
    bg_color : Tuple[int, int, int]
        Background color in BGR
    invert : bool
        Invert brightness mapping
    mode : str
        Character mode: "standard", "extended", "binary", "blocks", etc.
    use_color : bool
        If True, use original frame colors for characters
    output_width : int
        Final output image width (default 1280 for 16:9)
    output_height : int
        Final output image height (default 720 for 16:9)
    
    Returns:
    --------
    np.ndarray
        Image with rendered ASCII characters at native output resolution
    """
    if frame is None or frame.size == 0:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Character sets for different modes
    if mode == "binary":
        ascii_chars_full = "01"
    elif mode == "blocks":
        ascii_chars_full = "█▓▒░ "
    elif mode == "extended":
        ascii_chars_full = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    elif mode == "highcontrast":
        ascii_chars_full = "█▓▒░■□●○◆◇★☆+:. "
    elif mode == "simple_hd":
        ascii_chars_full = "@#W$9876543210?!abc;:+=-,._ "
    else:  # standard
        ascii_chars_full = "@%#*+=-:. "
    
    if invert:
        ascii_chars = ascii_chars_full[::-1]
    else:
        ascii_chars = ascii_chars_full
    
    num_chars = len(ascii_chars)
    
    # Convert to grayscale for brightness mapping
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color_frame = frame.copy()
    else:
        gray = frame
        color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Calculate character size to fit output dimensions exactly
    # Target: fill the entire output width with 'width' number of columns
    char_width = max(4, output_width // width)
    char_height = int(char_width * 1.8)  # Character aspect ratio ~1.8:1 (height:width)
    char_height = max(6, char_height)
    
    # Recalculate to ensure we fill the ENTIRE output width
    # Adjust number of columns to exactly fill the width
    actual_columns = output_width // char_width
    
    # Calculate how many rows fit in the output
    ascii_height = output_height // char_height
    ascii_height = max(1, ascii_height)
    
    # Resize input frame to match ASCII grid (stretches to fill)
    small_gray = cv2.resize(gray, (actual_columns, ascii_height), interpolation=cv2.INTER_AREA)
    small_color = cv2.resize(color_frame, (actual_columns, ascii_height), interpolation=cv2.INTER_AREA)
    
    # Font settings - calculate font scale based on character size
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font to fit character cell
    font_scale = char_height / 30.0  # Approximate scaling factor
    font_scale = max(0.2, min(1.0, font_scale))  # Clamp between 0.2 and 1.0
    thickness = 1
    
    # Create output image at exact output resolution
    img = np.full((output_height, output_width, 3), bg_color, dtype=np.uint8)
    
    # Draw ASCII characters - render directly at output resolution
    for row in range(ascii_height):
        for col in range(actual_columns):
            brightness = int(small_gray[row, col])
            
            # Map brightness to character index
            char_idx = min(num_chars - 1, int(brightness * num_chars / 256))
            char = ascii_chars[char_idx]
            
            # Calculate position (centered in cell)
            x = col * char_width + 1
            y = (row + 1) * char_height - 2
            
            # Ensure we don't draw outside bounds
            if x >= output_width or y >= output_height:
                continue
            
            # Determine color
            if use_color:
                b, g, r = small_color[row, col]
                color = (int(min(255, b * 1.2)), int(min(255, g * 1.2)), int(min(255, r * 1.2)))
            else:
                color = fg_color
            
            # Draw character
            cv2.putText(img, char, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img


def get_brightness_stats(frame: np.ndarray) -> dict:
    """
    Get brightness statistics for a frame.
    
    Useful for adjusting ASCII conversion parameters dynamically.
    
    Parameters:
    -----------
    frame : np.ndarray
        Input frame
    
    Returns:
    --------
    dict
        Dictionary with 'mean', 'std', 'min', 'max' brightness values
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    return {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': int(np.min(gray)),
        'max': int(np.max(gray))
    }
