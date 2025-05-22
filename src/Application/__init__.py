"""
Application layer for Realtime_mlx_STT.

This package provides the public-facing APIs and server components
for the speech-to-text library.

Components:
- Server: FastAPI-based HTTP/WebSocket server for remote access
- Facade: (Planned) Simplified Python API for direct library usage
"""

from .Server.ServerModule import ServerModule

__all__ = [
    'ServerModule'
]