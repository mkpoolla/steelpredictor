"""
Polyfill for the imghdr module which was removed in Python 3.13

This provides the essential functionality needed by Streamlit's image handling
"""
import io
import os

# Dictionary of all image test functions
tests = []

def what(file, h=None):
    """
    Determine the type of image contained in a file or memory buffer.
    
    Args:
        file: A filename (string) or a file object open for reading in binary mode
        h: The first few bytes of the file, if already read
        
    Returns:
        A string describing the image type if recognized, or None if not recognized
    """
    if h is None:
        # Try to open the file if it's a string
        if isinstance(file, str):
            try:
                with open(file, 'rb') as f:
                    h = f.read(32)
            except (IOError, OSError):
                return None
        else:
            # Try to read from file object
            try:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
            except (IOError, OSError):
                return None

    for test in tests:
        result = test(h, file)
        if result:
            return result

    # Explicit checks for the most common formats
    # Check for jpeg
    if h[0:2] == b'\xff\xd8':
        return 'jpeg'
    
    # Check for png
    if h[0:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    
    # Check for gif
    if h[0:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    
    # Check for bmp
    if h[0:2] == b'BM':
        return 'bmp'
    
    # Check for webp
    if h[0:4] == b'RIFF' and h[8:12] == b'WEBP':
        return 'webp'
        
    return None

def test_jpeg(h, f):
    """Test for JPEG data"""
    if h[0:2] == b'\xff\xd8':
        return 'jpeg'
    return None

def test_png(h, f):
    """Test for PNG data"""
    if h[0:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    return None

def test_gif(h, f):
    """Test for GIF data"""
    if h[0:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    return None

def test_bmp(h, f):
    """Test for BMP data"""
    if h[0:2] == b'BM':
        return 'bmp'
    return None

def test_webp(h, f):
    """Test for WebP data"""
    if h[0:4] == b'RIFF' and h[8:12] == b'WEBP':
        return 'webp'
    return None

# Register all the test functions
tests.append(test_jpeg)
tests.append(test_png)
tests.append(test_gif)
tests.append(test_bmp)
tests.append(test_webp)
