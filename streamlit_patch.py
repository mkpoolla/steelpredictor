#!/usr/bin/env python3
"""
Patch script for Streamlit to handle missing imghdr module in Python 3.13
This script uses Python's import system to provide a virtual imghdr module
"""
import sys
import os
from importlib.abc import MetaPathFinder, Loader
from importlib import machinery
from types import ModuleType
from pathlib import Path

# Project root for finding our custom modules
PROJECT_ROOT = Path(__file__).parent.absolute()

class ImghdrLoader(Loader):
    """Custom loader for the imghdr module"""
    
    def create_module(self, spec):
        """Create a new imghdr module with the necessary functions"""
        module = ModuleType("imghdr")
        module.__file__ = __file__
        module.__package__ = ""
        module.__path__ = []
        module.__loader__ = self
        module.__spec__ = spec
        
        # Add the core functions Streamlit needs
        def what(file, h=None):
            """Determine image type - simple implementation for Streamlit"""
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
            
            # Check for jpeg
            if h[0:2] == b'\xff\xd8':
                return 'jpeg'
            
            # Check for png
            if h[0:8] == b'\x89PNG\r\n\x1a\n':
                return 'png'
            
            # Check for gif
            if h[0:6] in (b'GIF87a', b'GIF89a'):
                return 'gif'
                
            return None
            
        # Add functions to the module
        module.what = what
        module.test_jpeg = lambda h, f: h[0:2] == b'\xff\xd8'
        module.test_png = lambda h, f: h[0:8] == b'\x89PNG\r\n\x1a\n'
        module.test_gif = lambda h, f: h[0:6] in (b'GIF87a', b'GIF89a')
        
        return module
        
    def exec_module(self, module):
        """Nothing needs to be executed since we created everything in create_module"""
        pass

class ImghdrFinder(MetaPathFinder):
    """Custom finder for the imghdr module"""
    
    def find_spec(self, fullname, path, target=None):
        """Find the imghdr module and return a spec for loading it"""
        if fullname == 'imghdr':
            loader = ImghdrLoader()
            return machinery.ModuleSpec(fullname, loader)
        return None

# Install our custom finder at the beginning of sys.meta_path
sys.meta_path.insert(0, ImghdrFinder())

# Now run the actual Streamlit app
if __name__ == '__main__':
    # Add project root to Python path
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Import and run the streamlit app
    from dashboard.app import main
    main()
