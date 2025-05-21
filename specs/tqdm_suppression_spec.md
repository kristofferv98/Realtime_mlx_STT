# tqdm Progress Bar Suppression Specification

## Problem Statement

The project uses tqdm for displaying progress bars in various components, particularly during model loading and file fetching operations. When running examples like `wake_word_detection.py`, these progress bars appear in the console output, creating visual noise and disrupting the cleaner user experience we want to provide.

Current issues:
- Progress bars appear during Hugging Face model downloads
- Progress bars appear during "Fetching files" operations
- Our current tqdm patching approach is inconsistent and causes compatibility issues

## Objectives

1. Implement a global, consistent solution to suppress tqdm progress bars across the entire application
2. Maintain functionality of the underlying operations
3. Ensure compatibility with all dependencies that use tqdm
4. Allow selective enabling/disabling of progress bars through configuration

## Technical Approach

### Option 1: Centralized tqdm Configuration Module

Create a dedicated module in the Infrastructure layer to configure tqdm globally:

```
src/Infrastructure/ProgressBar/
    __init__.py
    ProgressBarManager.py  # Main configuration module
    tqdm_config.py         # Configuration constants
```

### Option 2: Add to Existing Logging Infrastructure

Extend the existing LoggingModule to also control progress bar visibility:

```
src/Infrastructure/Logging/
    LoggingModule.py       # Add progress bar control methods
    ProgressBarConfig.py   # New configuration file
```

## Implementation Plan

### Core Implementation (Preferred Approach)

We will implement a dedicated ProgressBarManager in the Infrastructure layer that will:

1. Provide global configuration of tqdm settings
2. Use tqdm's built-in disable parameter to suppress bars
3. Use environment variables to control behavior
4. Offer runtime configuration through the application API

### Technical Details

1. **Global tqdm Configuration**:
   ```python
   # In ProgressBarManager.py
   import tqdm
   import os
   
   class ProgressBarManager:
       _initialized = False
       _disabled = False
       
       @classmethod
       def initialize(cls, disabled=None):
           """Initialize progress bar settings globally."""
           # Check environment variable first
           env_disabled = os.environ.get("DISABLE_PROGRESS_BARS", "").lower() in ("true", "1", "yes")
           
           # Priority: explicit parameter > environment variable > default (enabled)
           cls._disabled = disabled if disabled is not None else env_disabled
           
           # Configure tqdm globally
           tqdm.tqdm.set_lock(tqdm.tqdm.get_lock())  # Ensure lock is initialized
           tqdm.tqdm.disable = cls._disabled
           
           # Disable HuggingFace progress bars if needed
           if cls._disabled:
               os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
           
           cls._initialized = True
           return cls._disabled
       
       @classmethod
       def disable(cls):
           """Disable progress bars globally."""
           if not cls._initialized:
               cls.initialize()
           cls._disabled = True
           tqdm.tqdm.disable = True
           os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
       
       @classmethod
       def enable(cls):
           """Enable progress bars globally."""
           if not cls._initialized:
               cls.initialize()
           cls._disabled = False
           tqdm.tqdm.disable = False
           os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
       
       @classmethod
       def is_disabled(cls):
           """Check if progress bars are disabled."""
           if not cls._initialized:
               cls.initialize()
           return cls._disabled
   ```

2. **Integration with Examples**:
   ```python
   # In examples/wake_word_detection.py and other examples
   from src.Infrastructure.ProgressBar.ProgressBarManager import ProgressBarManager
   
   # Initialize at the beginning of the script
   ProgressBarManager.initialize(disabled=args.quiet)
   ```

3. **Environment Variable Support**:
   Add to shell scripts and documentation:
   ```bash
   # Disable progress bars
   export DISABLE_PROGRESS_BARS=true
   
   # Run example
   python -m examples.wake_word_detection
   ```

4. **Hugging Face Hub Integration**:
   Ensure Hugging Face Hub progress bars are also disabled:
   ```python
   # Will be handled by ProgressBarManager.initialize()
   os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
   ```

## Implementation Tasks

1. Create the ProgressBarManager module in Infrastructure
2. Update all examples to use the ProgressBarManager
3. Update any direct tqdm usage in the codebase to respect the global configuration
4. Add command-line arguments to examples for controlling progress bar visibility
5. Add documentation on progress bar control in README.md
6. Update the LoggingModule to be aware of ProgressBarManager settings

## Testing Plan

1. Verify progress bars are hidden when disabled:
   - Test with environment variable
   - Test with explicit disable call
   - Test with command-line arguments
   
2. Verify progress bars show when enabled:
   - Test with default settings
   - Test with explicit enable call
   
3. Test with various examples:
   - wake_word_detection.py
   - continuous_transcription.py
   - transcribe_file.py

## Benefits

1. Consistent approach across the entire application
2. No need for patching or monkey-patching tqdm
3. Easier maintenance and fewer compatibility issues
4. User-configurable behavior
5. Cleaner console output during normal operation

## Migration Plan

1. Remove existing tqdm patching code in wake_word_detection.py and other examples
2. Implement ProgressBarManager
3. Update examples to use ProgressBarManager
4. Test thoroughly to ensure no progress bars appear in quiet mode