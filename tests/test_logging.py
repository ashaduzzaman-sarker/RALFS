import logging
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from ralfs.core.logging import get_logger, setup_logging, RALFS_FORMATTER

class TestGetLogger:
    def test_get_logger_default(self):
        """Test getting default logger with a name."""
        logger = get_logger("default_test_logger") # Added required 'name' argument
        assert logger.name == "default_test_logger"
        assert logger.level == logging.INFO
        assert logger.propagate is True
        # Check that it has a handler
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert logger.handlers[0].formatter is not None

    def test_get_logger_custom_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("my_custom_logger")
        assert logger.name == "my_custom_logger"
        assert logger.level == logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_get_logger_custom_level(self):
        """Test getting logger with custom level."""
        logger = get_logger("debug_logger", logging.DEBUG)
        assert logger.name == "debug_logger"
        assert logger.level == logging.DEBUG
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_logger_has_handlers(self):
        """Test that logger has at least one handler."""
        logger = get_logger("handler_test")
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_logging_works(self, caplog):
        """Test that logger actually logs messages and caplog captures them."""
        logger = get_logger("ralfs.test.log")
        with caplog.at_level(logging.INFO):
            logger.info("Test message from caplog test")
            assert "Test message from caplog test" in caplog.text


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def reset_logging_handlers(self):
        # This fixture will run before and after each test in this class
        # to ensure a clean slate for logging setup tests.
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        
        # Clear handlers from root and all specific loggers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for name, logger_obj in logging.root.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                for handler in logger_obj.handlers[:]:
                    logger_obj.removeHandler(handler)

        yield
        
        # Restore original handlers and level
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.level = original_level
        
        # Clear handlers from specific loggers again if they were added during tests
        for name, logger_obj in logging.root.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                for handler in logger_obj.handlers[:]:
                    logger_obj.removeHandler(handler)


    def test_setup_logging_console_only(self):
        """Test setup with console output only."""
        setup_logging(log_level="INFO") # Changed 'level' to 'log_level', removed console_output=True
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        assert not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers) # No file handler

    def test_setup_logging_with_file(self, tmp_path):
        """Test setup with file output."""
        log_file = tmp_path / "test.log"
        setup_logging(log_level="DEBUG", log_file=log_file) # Changed 'level' to 'log_level'
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        assert any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        assert log_file.exists()
        # Verify file content
        root_logger.info("File log test message")
        with open(log_file, 'r') as f:
            content = f.read()
            assert "File log test message" in content

    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that setup_logging creates the log directory if it doesn't exist."""
        log_dir = tmp_path / "logs"
        log_file = log_dir / "app.log"
        setup_logging(log_file=log_file)
        assert log_dir.is_dir()
        assert log_file.exists()

    def test_setup_logging_suppresses_verbose_libraries(self):
        """Test that verbose libraries are suppressed."""
        setup_logging(log_level="INFO") # Changed 'level' to 'log_level'
        assert logging.getLogger("transformers").level == logging.WARNING
        assert logging.getLogger("huggingface_hub").level == logging.WARNING
        assert logging.getLogger("datasets").level == logging.WARNING
        assert logging.getLogger("sentence_transformers").level == logging.WARNING
