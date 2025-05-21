"""
Simple utility to disable tqdm progress bars.

This module sets environment variables and also attempts to monkey patch
tqdm to disable progress bars consistently.
"""

import os
import sys
import importlib.util

# Set environment variables to disable progress bars globally
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Try to monkey patch tqdm directly
def patch_tqdm():
    """Try to monkey patch tqdm to completely disable progress bars."""
    try:
        # Check if tqdm is already imported
        if 'tqdm' in sys.modules:
            # Get the tqdm module
            tqdm_module = sys.modules['tqdm']
            
            # Create the original tqdm class if needed for reference
            if not hasattr(tqdm_module, '_original_tqdm') and hasattr(tqdm_module, 'tqdm'):
                tqdm_module._original_tqdm = tqdm_module.tqdm
            
            # Create a dummy tqdm function
            def _dummy_tqdm(iterable=None, *args, **kwargs):
                if iterable is not None:
                    return iterable
                return DummyTqdm()
            
            # Copy attributes from original tqdm
            if hasattr(tqdm_module, '_original_tqdm'):
                for attr_name in dir(tqdm_module._original_tqdm):
                    if not attr_name.startswith('__') and not hasattr(_dummy_tqdm, attr_name):
                        setattr(_dummy_tqdm, attr_name, getattr(tqdm_module._original_tqdm, attr_name))
            
            # Replace tqdm with our dummy version
            tqdm_module.tqdm = _dummy_tqdm
            
            # Also try to patch tqdm.auto
            if hasattr(tqdm_module, 'auto') and hasattr(tqdm_module.auto, 'tqdm'):
                tqdm_module.auto.tqdm = _dummy_tqdm
        
        # Try to import tqdm if it's not already imported
        else:
            # Find the tqdm module spec without importing it
            spec = importlib.util.find_spec('tqdm')
            if spec:
                # We could monkey patch tqdm here, but it's better to let the environment 
                # variables do their work before tqdm is imported
                pass
    except Exception as e:
        # If anything goes wrong, just continue - environment variables will still work
        pass

# Dummy tqdm class for returning when needed
class DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args, **kwargs):
        pass
    
    def update(self, *args, **kwargs):
        pass
    
    def close(self, *args, **kwargs):
        pass
    
    def set_description(self, *args, **kwargs):
        pass
    
    def set_postfix(self, *args, **kwargs):
        pass

# Patch tqdm when this module is imported
patch_tqdm()

def disable():
    """
    Explicitly disable progress bars.
    
    This function can be called to ensure tqdm progress bars are disabled.
    It sets environment variables and attempts to monkey patch tqdm if loaded.
    """
    # Environment variables are already set at import time
    # Attempt to patch tqdm again in case it was imported after this module
    patch_tqdm()