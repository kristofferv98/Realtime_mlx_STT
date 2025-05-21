"""
ClearVadPreSpeechBufferCommand for Voice Activity Detection.

This command instructs the VoiceActivityHandler to clear its
internal pre-speech audio buffer.
"""
from dataclasses import dataclass
from src.Core.Commands.command import Command

@dataclass
class ClearVadPreSpeechBufferCommand(Command):
    """
    Command to clear the VAD's pre-speech buffer.
    Typically used after a wake word is detected to ensure transcription
    starts with audio captured post-wake-word.
    """
    pass