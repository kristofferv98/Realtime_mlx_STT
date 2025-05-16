"""
TranscriptionProcessManager for process isolation of the transcription engine.

This module manages running transcription engines in separate Python processes
for better stability, resource management, and isolation.
"""

import logging
import multiprocessing
from multiprocessing.connection import Connection
import os
import time
from typing import Dict, Any, Optional, Union

# Timeout constants
STARTUP_TIMEOUT = 10.0  # Seconds to wait for process startup
COMMAND_TIMEOUT = 120.0  # Seconds to wait for command response (increased for large files)
SHUTDOWN_TIMEOUT = 10.0  # Seconds to wait for clean shutdown


class TranscriptionProcessManager:
    """
    Manages the transcription process in a separate Python process.
    
    This class is responsible for:
    1. Creating and managing a separate process for transcription
    2. Communicating with the process via pipes
    3. Handling the process lifecycle (start, stop, monitor)
    4. Providing a thread-safe interface for accessing transcription functionality
    """
    
    def __init__(self):
        """Initialize the process manager."""
        self.logger = logging.getLogger(__name__)
        self.process = None
        self.parent_pipe = None
        self.child_pipe = None
    
    def start(self, engine_type: str = "mlx_whisper", engine_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start the transcription process with the specified engine.
        
        Args:
            engine_type: Type of transcription engine to use
            engine_config: Configuration for the transcription engine
            
        Returns:
            bool: True if process started successfully
        """
        self.logger.info(f"Starting transcription process with engine_type={engine_type}")
        
        # Create pipes for communication
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()
        
        # Create and start the process
        self.process = multiprocessing.Process(
            target=self._run_transcription_process,
            args=(self.child_pipe, engine_type, engine_config or {}),
            daemon=True
        )
        self.process.start()
        
        # Verify process started correctly
        if not self.parent_pipe.poll(STARTUP_TIMEOUT):
            self.logger.error("Timeout waiting for transcription process to start")
            self.stop()
            return False
        
        response = self.parent_pipe.recv()
        success = response.get('success', False)
        
        if success:
            self.logger.info("Transcription process started successfully")
        else:
            self.logger.error(f"Failed to start transcription process: {response.get('error', 'Unknown error')}")
            self.stop()
        
        return success
    
    def transcribe(self, 
                  audio_data: Any, 
                  is_first_chunk: bool = False, 
                  is_last_chunk: bool = False, 
                  options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send audio data to transcription process.
        
        Args:
            audio_data: Audio data to transcribe
            is_first_chunk: Whether this is the first chunk of audio
            is_last_chunk: Whether this is the last chunk of audio
            options: Additional options for transcription
            
        Returns:
            Dict[str, Any]: Transcription result or error
        """
        if not self.is_running():
            return {"error": "Transcription process not running"}
        
        # Send transcription request
        self.parent_pipe.send({
            'command': 'TRANSCRIBE',
            'audio_data': audio_data,
            'is_first_chunk': is_first_chunk,
            'is_last_chunk': is_last_chunk,
            'options': options or {}
        })
        
        # Wait for response
        if self.parent_pipe.poll(COMMAND_TIMEOUT):
            return self.parent_pipe.recv()
        else:
            self.logger.warning("Timeout waiting for transcription result")
            return {"error": "Transcription timeout"}
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the transcription engine.
        
        Args:
            config: New configuration for the engine
            
        Returns:
            bool: True if configuration was successful
        """
        if not self.is_running():
            self.logger.warning("Cannot configure - transcription process not running")
            return False
        
        # Send configuration request
        self.parent_pipe.send({
            'command': 'CONFIGURE',
            'config': config
        })
        
        # Wait for response
        if self.parent_pipe.poll(STARTUP_TIMEOUT):
            response = self.parent_pipe.recv()
            return response.get('success', False)
        else:
            self.logger.warning("Timeout waiting for configuration response")
            return False
    
    def stop(self) -> bool:
        """
        Stop the transcription process.
        
        Returns:
            bool: True if process was successfully stopped
        """
        if self.is_running():
            self.logger.info("Stopping transcription process")
            
            try:
                # Send shutdown command
                self.parent_pipe.send({'command': 'SHUTDOWN'})
                
                # Wait for process to terminate
                self.process.join(SHUTDOWN_TIMEOUT)
                
                # Force terminate if necessary
                if self.process.is_alive():
                    self.logger.warning("Process did not terminate gracefully, forcing termination")
                    self.process.terminate()
                    self.process.join(2.0)
                
                # Clean up
                self.parent_pipe.close()
                self.child_pipe.close()
                
            except Exception as e:
                self.logger.error(f"Error stopping transcription process: {e}")
            
            finally:
                self.process = None
                self.parent_pipe = None
                self.child_pipe = None
        
        return True
    
    def is_running(self) -> bool:
        """
        Check if transcription process is running.
        
        Returns:
            bool: True if process is running
        """
        return self.process is not None and self.process.is_alive()
    
    @staticmethod
    def _run_transcription_process(pipe: Connection, 
                                  engine_type: str, 
                                  engine_config: Dict[str, Any]) -> None:
        """
        Process function that runs the transcription engine.
        
        This function is run in a separate process and communicates with the
        main process via the provided pipe.
        
        Args:
            pipe: Pipe for communication with main process
            engine_type: Type of transcription engine to use
            engine_config: Configuration for the transcription engine
        """
        logger = logging.getLogger("TranscriptionProcess")
        logger.info(f"Starting transcription process with engine_type={engine_type}")
        
        # Create and initialize the appropriate engine
        engine = None
        
        try:
            # Initialize the appropriate engine based on type
            if engine_type == "mlx_whisper":
                # Import here to avoid circular imports
                from src.Features.Transcription.Engines.MlxWhisperEngine import MlxWhisperEngine
                logger.info(f"Creating MlxWhisperEngine with config: {engine_config}")
                
                # Ensure critical config values have defaults
                if 'model_name' not in engine_config:
                    engine_config['model_name'] = "whisper-large-v3-turbo"
                    
                engine = MlxWhisperEngine(**(engine_config or {}))
                
            elif engine_type == "remote":
                # Placeholder for remote transcription engine
                from src.Features.Transcription.Engines.RemoteTranscriptionEngine import RemoteTranscriptionEngine
                engine = RemoteTranscriptionEngine(**(engine_config or {}))
                
            else:
                logger.error(f"Unknown engine type: {engine_type}")
                pipe.send({'success': False, 'error': f"Unknown engine type: {engine_type}"})
                return
            
            # Initialize the engine with retry logic
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
                logger.info(f"Initializing engine (attempt {retry_count + 1}/{max_retries + 1})...")
                
                if engine and engine.start():
                    # Signal ready on success
                    logger.info("Engine initialized successfully")
                    pipe.send({'success': True})
                    break  # Exit retry loop on success
                    
                # Increment retry counter if initialization failed
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Engine initialization failed, retrying ({retry_count}/{max_retries})...")
                    time.sleep(1)  # Wait before retrying
            
            # Check if initialization succeeded
            if not engine or not engine.is_running():
                logger.error("Failed to initialize engine after retries")
                pipe.send({'success': False, 'error': "Failed to initialize engine after retries"})
                return
                
            # Process commands until shutdown
            while True:
                if pipe.poll(0.1):
                    command = pipe.recv()
                    
                    if command['command'] == 'TRANSCRIBE':
                        # Process audio data
                        try:
                            # For batch transcription of complete files
                            if command.get('is_first_chunk', False) and command.get('is_last_chunk', False):
                                logger.info("Processing full batch transcription")
                                
                                # Direct transcribe call for full audio
                                try:
                                    # Check if audio is a file path or data
                                    if isinstance(command['audio_data'], str) and os.path.exists(command['audio_data']):
                                        logger.info(f"Transcribing file: {command['audio_data']}")
                                        engine.transcribe(command['audio_data'])
                                    else:
                                        # Process audio data
                                        engine.transcribe(command['audio_data'])
                                    
                                    # Directly wait for result with longer timeout
                                    timeout = command.get('options', {}).get('timeout', 30.0)  # Longer timeout for full files
                                    result = engine.get_result(timeout=timeout)
                                    
                                    # Send result
                                    if result:
                                        pipe.send(result)
                                    else:
                                        # If no result, generate error response
                                        pipe.send({
                                            "text": "",
                                            "is_final": True,
                                            "confidence": 0.0,
                                            "status": "no_result",
                                            "error": "No transcription result available"
                                        })
                                except Exception as transcribe_error:
                                    logger.error(f"Error in direct transcription: {transcribe_error}")
                                    pipe.send({
                                        "text": "",
                                        "is_final": True,
                                        "confidence": 0.0,
                                        "status": "error",
                                        "error": f"Transcription error: {str(transcribe_error)}"
                                    })
                            else:
                                # Regular streaming mode
                                # Add audio chunk to engine
                                engine.add_audio_chunk(
                                    command['audio_data'],
                                    command.get('is_last_chunk', False)
                                )
                                
                                # Get result (blocks until available or timeout)
                                timeout = command.get('options', {}).get('timeout', 5.0)
                                result = engine.get_result(timeout=timeout)
                                
                                # Send result back with better fallback
                                if result is None:
                                    # Provide a well-formed empty result
                                    pipe.send({
                                        "text": "",
                                        "is_final": command.get('is_last_chunk', False),
                                        "confidence": 0.0,
                                        "status": "no_result",
                                        "error": "No result available"
                                    })
                                else:
                                    pipe.send(result)
                                
                        except Exception as e:
                            logger.exception(f"Error during transcription: {e}")
                            pipe.send({"error": str(e)})
                            
                    elif command['command'] == 'CONFIGURE':
                        # Configure the engine
                        try:
                            # The configure method varies between engine implementations
                            if hasattr(engine, 'configure'):
                                success = engine.configure(command['config'])
                                pipe.send({'success': success})
                            else:
                                logger.warning("Engine does not support runtime configuration")
                                pipe.send({'success': False, 'error': "Engine does not support runtime configuration"})
                                
                        except Exception as e:
                            logger.exception(f"Error configuring engine: {e}")
                            pipe.send({'success': False, 'error': str(e)})
                        
                    elif command['command'] == 'SHUTDOWN':
                        # Clean up and exit
                        logger.info("Received shutdown command")
                        break
                    
                    else:
                        logger.warning(f"Unknown command: {command['command']}")
                        pipe.send({'error': f"Unknown command: {command['command']}"})
            else:
                # Signal initialization failure
                error_msg = "Failed to initialize transcription engine"
                logger.error(error_msg)
                pipe.send({'success': False, 'error': error_msg})
                
        except Exception as e:
            # Signal error
            logger.exception(f"Error in transcription process: {e}")
            pipe.send({'success': False, 'error': str(e)})
            
        finally:
            # Clean up
            logger.info("Cleaning up transcription process")
            if engine:
                try:
                    engine.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up engine: {e}")
            
            try:
                pipe.close()
            except Exception as e:
                logger.error(f"Error closing pipe: {e}")