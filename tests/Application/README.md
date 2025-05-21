# Application Layer Tests

This directory contains tests for the Application layer components of the Realtime_mlx_STT system.

## Server Tests

The Server tests validate the functionality of the server-based architecture, including:

- Server registration and initialization
- Configuration management
- WebSocket communication
- API endpoints
- Integration with the command/event system

## Test Structure

The tests are organized to mirror the structure of the Application layer:

```
tests/Application/
├── Server/
│   ├── Configuration/          # Tests for configuration components
│   │   ├── test_profile_manager.py
│   │   └── test_server_config.py
│   ├── Controllers/            # Tests for API controllers
│   │   ├── test_system_controller.py
│   │   └── test_transcription_controller.py
│   ├── WebSocket/              # Tests for WebSocket components
│   │   └── test_websocket_manager.py
│   └── test_server_module.py   # Tests for ServerModule
├── README.md
└── run_tests.py                # Test runner script
```

## Running Tests

To run all Application tests:

```bash
cd tests/Application
./run_tests.py
```

To run tests for a specific component:

```bash
./run_tests.py -c Server
```

To run a specific test:

```bash
./run_tests.py -t test_server_module
```

For verbose output:

```bash
./run_tests.py -v
```

## Test Dependencies

The tests require the following dependencies:

- fastapi
- uvicorn
- pytest
- httpx (for FastAPI TestClient)

Make sure these are installed before running the tests:

```bash
uv sync pyproject.toml
```

## Test Types

The Application tests include:

1. **Unit Tests**: Testing individual components in isolation
2. **Integration Tests**: Testing components working together
3. **API Tests**: Testing API endpoints using FastAPI's TestClient

## Mocking

Many tests use mocking to isolate the component being tested:

- `unittest.mock.MagicMock` for synchronous mocks
- `unittest.mock.AsyncMock` for asynchronous mocks
- `unittest.mock.patch` for patching dependencies