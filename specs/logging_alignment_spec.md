# Logging Alignment Specification for Realtime_mlx_STT

## 1. Introduction

This specification defines the plan for aligning the logging system across all components of the Realtime_mlx_STT codebase, based on the analysis in `specs/logging_analysis.md`. The goal is to ensure consistent logger acquisition, standardized namespace conventions, and centralized log level control throughout the application.

## 2. Current Status Overview

The codebase implements a robust logging infrastructure in `src/Infrastructure/Logging/` that follows sound design principles. However, there are inconsistencies in how loggers are obtained and used across different components:

1. **Core modules** use explicit standardized names:
   ```python
   # src/Core/Commands/command_dispatcher.py
   logger = logging.getLogger("realtimestt.core.commands")
   ```

2. **Feature modules** typically use the Python module path directly:
   ```python
   # src/Features/VoiceActivityDetection/VadModule.py
   logger = logging.getLogger(__name__)
   ```

3. **Direct log level setting** occurs in some components:
   ```python
   # src/Features/VoiceActivityDetection/Detectors/CombinedVadDetector.py
   self.logger.setLevel(logging.INFO)
   ```

4. **Logging from model classes** that should ideally be simple data carriers:
   ```python
   # src/Features/VoiceActivityDetection/Commands/ConfigureVadCommand.py
   import logging
   logging.getLogger(__name__).warning(...)
   ```

## 3. Issues to Address

1. **Inconsistent Logger Acquisition**: When loggers are obtained via `logging.getLogger(__name__)`, they are not properly standardized to the expected namespace convention (`realtimestt.features.audiocapture`). As a result, feature-specific log level settings in `LoggingConfig.feature_levels` cannot correctly target these loggers.

2. **Direct Log Level Setting**: Direct calls to `logger.setLevel()` bypass the centralized configuration system, creating inconsistencies in log level management.

3. **Logging from Model Classes**: Some command and event model classes perform logging, which couples them to the logging system and makes them more than simple data carriers.

4. **Core Component Log Levels**: The current configuration system has no specific mechanism for managing core component log levels separately from the root level.

## 4. Alignment Plan

### 4.1 Logger Acquisition Standardization

#### Files to Modify

All feature modules that use direct `logging.getLogger(__name__)` calls must be updated to use `LoggingModule.get_logger(__name__)` instead. 

First, run a comprehensive search to identify all instances:

```bash
grep -r "logging.getLogger(__name__)" --include="*.py" /Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/src/Features/
```

Based on the preliminary analysis, these files will need modification:

1. **VoiceActivityDetection**:
   - `src/Features/VoiceActivityDetection/VadModule.py`
   - `src/Features/VoiceActivityDetection/Detectors/CombinedVadDetector.py`
   - `src/Features/VoiceActivityDetection/Detectors/SileroVadDetector.py`
   - `src/Features/VoiceActivityDetection/Detectors/WebRtcVadDetector.py`
   - `src/Features/VoiceActivityDetection/Handlers/VoiceActivityHandler.py`

2. **WakeWordDetection**:
   - `src/Features/WakeWordDetection/WakeWordModule.py`
   - `src/Features/WakeWordDetection/Detectors/PorcupineWakeWordDetector.py`
   - `src/Features/WakeWordDetection/Handlers/WakeWordCommandHandler.py`

3. **Transcription**:
   - `src/Features/Transcription/TranscriptionModule.py`
   - `src/Features/Transcription/Engines/DirectMlxWhisperEngine.py`
   - `src/Features/Transcription/Engines/OpenAITranscriptionEngine.py`
   - `src/Features/Transcription/Engines/DirectTranscriptionManager.py`
   - `src/Features/Transcription/Handlers/TranscriptionCommandHandler.py`

4. **AudioCapture**:
   - `src/Features/AudioCapture/AudioCaptureModule.py`
   - `src/Features/AudioCapture/Handlers/AudioCommandHandler.py`
   - `src/Features/AudioCapture/Providers/FileAudioProvider.py`
   - `src/Features/AudioCapture/Providers/PyAudioInputProvider.py`

#### Implementation Pattern

Replace:
```python
import logging
logger = logging.getLogger(__name__)
```

With:
```python
from src.Infrastructure.Logging import LoggingModule
logger = LoggingModule.get_logger(__name__)
```

For class instance initialization, replace:
```python
self.logger = logging.getLogger(__name__)
```

With:
```python
self.logger = LoggingModule.get_logger(__name__)
```

### 4.2 Remove Direct Log Level Setting

#### Files to Modify

Remove direct log level setting from:

1. `src/Features/VoiceActivityDetection/Detectors/CombinedVadDetector.py` (line 70):
   ```python
   self.logger.setLevel(logging.INFO)
   ```

The log level should be controlled exclusively through the configuration system:
- At global level through `LoggingModule.initialize()`
- At runtime through `LoggingModule.set_level()`

Search for other instances where logger levels might be directly set:
```bash
grep -r "setLevel" --include="*.py" /Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/src/Features/
```

### 4.3 Refactor Logging from Model Classes

#### Files to Modify

1. `src/Features/VoiceActivityDetection/Commands/ConfigureVadCommand.py` (lines 68-73):
   ```python
   import logging
   logging.getLogger(__name__).warning(
       f"Pre-speech buffer size is very small ({self.pre_speech_buffer_size}). "
       f"This may not capture enough audio before speech detection. "
       f"Recommended minimum size is 16 chunks (~0.5 seconds)."
   )
   ```

#### Implementation Pattern

Option 1 (preferred): Move validation and logging to the command handler:
```python
# In VoiceActivityHandler.py
def handle(self, command: ConfigureVadCommand) -> bool:
    # Check for warning conditions
    if command.pre_speech_buffer_size < 16:
        self.logger.warning(
            f"Pre-speech buffer size is very small ({command.pre_speech_buffer_size}). "
            f"This may not capture enough audio before speech detection. "
            f"Recommended minimum size is 16 chunks (~0.5 seconds)."
        )
    # Rest of handler code...
```

Option 2 (if keeping in model): Use LoggingModule:
```python
# In ConfigureVadCommand.py
from src.Infrastructure.Logging import LoggingModule
# ...
if self.pre_speech_buffer_size < 16:
    logger = LoggingModule.get_logger(__name__)
    logger.warning(
        f"Pre-speech buffer size is very small ({self.pre_speech_buffer_size}). "
        f"This may not capture enough audio before speech detection. "
        f"Recommended minimum size is 16 chunks (~0.5 seconds)."
    )
```

Search for other model classes that might be performing logging:
```bash
grep -r "logging.getLogger" --include="*Command.py" /Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/src/Features/
grep -r "logging.getLogger" --include="*Event.py" /Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/src/Features/
```

### 4.4 Core Component Log Levels (Optional Enhancement)

> Note: Until this enhancement is implemented, core loggers (like `realtimestt.core.commands`) will be controlled by the `root_level` in LoggingConfig, not a specific core_levels entry.

#### Files to Modify

1. `src/Infrastructure/Logging/LoggingConfig.py`: Add core_levels field to LoggingConfig class:
   ```python
   # Add to fields in LoggingConfig class
   core_levels: Dict[str, LogLevel] = field(default_factory=dict)
   
   # Add getter method
   def get_core_level(self, core_name: str) -> LogLevel:
       """Get the log level for a specific core component."""
       return self.core_levels.get(core_name, self.root_level)
       
   # Add setter method
   def set_core_level(self, core_name: str, level: LogLevel) -> None:
       """Set the log level for a specific core component."""
       self.core_levels[core_name] = level
   ```

2. `src/Infrastructure/Logging/LoggingConfigurer.py`: Update `configure()` method to apply core levels:
   ```python
   # Add after feature_levels application in configure method
   # Configure core component loggers
   for core_name, level in config.core_levels.items():
       core_logger = logging.getLogger(f"{config.root_namespace}.core.{core_name}")
       core_logger.setLevel(level.value)
   ```

3. `src/Infrastructure/Logging/LoggingModule.py`: Add core level management:
   ```python
   # Add to initialize() method parameters
   core_levels: Dict[str, Union[str, LogLevel]] = None,
   
   # Add to initialize() method body, similar to feature_levels handling
   # Set core levels
   if core_levels:
       for core, level in core_levels.items():
           if isinstance(level, str):
               config.core_levels[core] = LogLevel.from_string(level)
           else:
               config.core_levels[core] = level
               
   # Add a new method
   @staticmethod
   def set_core_level(core_name: str, level: Union[str, LogLevel]) -> None:
       """
       Set the log level for a specific core component.
       
       Args:
           core_name: Name of the core component to set level for
           level: Log level to set
       """
       # Convert string level to LogLevel
       if isinstance(level, str):
           level_enum = LogLevel.from_string(level)
       else:
           level_enum = level
           
       LoggingConfigurer.set_core_level(core_name, level_enum)
   ```

4. `src/Infrastructure/Logging/LoggingConfigurer.py`: Add method to set core level:
   ```python
   @classmethod
   def set_core_level(cls, core_name: str, level: LogLevel) -> None:
       """
       Set the log level for a specific core component.
       
       Args:
           core_name: Name of the core component to set level for
           level: Log level to set
       """
       # If no configuration exists, use a default one
       if cls._current_config is None:
           cls.configure(LoggingConfig())
           
       # Update the configuration
       cls._current_config.set_core_level(core_name, level)
       
       # Update existing loggers
       core_namespace = f"{cls._current_config.root_namespace}.core.{core_name.lower()}"
       core_logger = logging.getLogger(core_namespace)
       core_logger.setLevel(level.value)
   ```

## 5. Implementation Priority

The recommended order of implementation:

1. **High Priority**:
   - Logger acquisition standardization (4.1)
   - Remove direct log level setting (4.2)

2. **Medium Priority**:
   - Refactor logging from model classes (4.3)

3. **Low Priority** (Optional Enhancement):
   - Add core component log level support (4.4)

## 6. Verification and Testing

After making these changes, verify correct functionality by:

1. **Testing Feature-Specific Log Levels**:
   - Configure a feature-specific log level (e.g., `AudioCapture: DEBUG`)
   - Verify that loggers within `src/Features/AudioCapture/*` output DEBUG messages
   - Verify that other feature loggers still follow the root log level
   - Before the changes, these loggers might have only responded to the root_level

2. **Testing Dynamic Log Level Changes**:
   - Use `LoggingModule.set_level("AudioCapture", "DEBUG")` at runtime
   - Verify that all loggers within the AudioCapture feature correctly change levels
   - Test both standardized feature name targets and direct logger name targets

3. **Testing README Examples**:
   - Ensure the example code in README.md's "Configuring Logging" section works correctly
   - Verify that all logging endpoints mentioned in the documentation function as expected

4. **Edge Cases**:
   - Test what happens when logging is used before `LoggingModule.initialize()` is called
   - Verify behavior when invalid log levels are provided as strings
   - Check behavior when mixing logger acquisition methods in a single class

## 7. Conclusion

By implementing these changes, we will ensure that the logging system works consistently across the entire codebase, with standardized namespace conventions and centralized log level control. This will improve maintainability, enable more precise log level control, and fully leverage the capabilities of the existing logging infrastructure.

Key benefits of this alignment include:
- Feature-specific log levels will work correctly for all components
- Logging configuration can be changed at runtime through a single interface
- The separation of concerns between model classes and handling logic is enhanced
- The logging system will be fully aligned with its design specifications