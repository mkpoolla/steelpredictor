"""
Simple polyfill for the missing imghdr module in Python 3.13+
"""

def what(file, h=None):
    """Dummy implementation that returns None for all files"""
    return None

# Other functions that might be needed
def test_jpeg(h, f):
    return None

def test_png(h, f):
    return None

def test_gif(h, f):
    return None
