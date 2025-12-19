"""Tests for ralfs.utils.io module."""

import pytest
import json
from pathlib import Path
from ralfs.utils.io import load_json, save_json, load_jsonl, save_jsonl


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "name": "RALFS",
        "version": "1.0.0",
        "tasks": ["retrieval", "generation", "evaluation"],
        "metrics": {"rouge": 0.85, "bertscore": 0.92},
    }


@pytest.fixture
def sample_jsonl_data():
    """Sample JSONL data for testing."""
    return [
        {"id": 1, "text": "First document", "label": "positive"},
        {"id": 2, "text": "Second document", "label": "negative"},
        {"id": 3, "text": "Third document", "label": "neutral"},
    ]


class TestLoadSaveJSON:
    """Tests for JSON loading and saving."""
    
    def test_save_and_load_json(self, temp_dir, sample_data):
        """Test basic JSON save and load."""
        file_path = temp_dir / "test.json"
        save_json(sample_data, file_path)
        
        assert file_path.exists()
        loaded_data = load_json(file_path)
        assert loaded_data == sample_data
    
    def test_save_json_creates_directories(self, temp_dir, sample_data):
        """Test that save_json creates parent directories."""
        file_path = temp_dir / "nested" / "dir" / "test.json"
        save_json(sample_data, file_path)
        
        assert file_path.exists()
        assert file_path.parent.exists()
    
    def test_load_nonexistent_file_raises_error(self, temp_dir):
        """Test that loading nonexistent file raises FileNotFoundError."""
        file_path = temp_dir / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_json(file_path)
    
    def test_load_invalid_json_raises_error(self, temp_dir):
        """Test that loading invalid JSON raises JSONDecodeError."""
        file_path = temp_dir / "invalid.json"
        file_path.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)


class TestLoadSaveJSONL:
    """Tests for JSONL loading and saving."""
    
    def test_save_and_load_jsonl(self, temp_dir, sample_jsonl_data):
        """Test basic JSONL save and load."""
        file_path = temp_dir / "test.jsonl"
        save_jsonl(sample_jsonl_data, file_path)
        
        assert file_path.exists()
        loaded_data = load_jsonl(file_path)
        assert loaded_data == sample_jsonl_data
    
    def test_load_jsonl_with_empty_lines(self, temp_dir):
        """Test that empty lines are skipped in JSONL."""
        file_path = temp_dir / "test.jsonl"
        file_path.write_text('{"id": 1}\n\n{"id": 2}\n\n')
        
        loaded_data = load_jsonl(file_path)
        assert len(loaded_data) == 2
        assert loaded_data[0]["id"] == 1
        assert loaded_data[1]["id"] == 2
    
    def test_load_jsonl_with_invalid_line(self, temp_dir):
        """Test that invalid lines are skipped with warning."""
        file_path = temp_dir / "test.jsonl"
        file_path.write_text('{"id": 1}\ninvalid line\n{"id": 2}')
        
        loaded_data = load_jsonl(file_path)
        # Should skip invalid line
        assert len(loaded_data) == 2
    
    def test_save_non_list_as_jsonl_raises_error(self, temp_dir, sample_data):
        """Test that saving non-list as JSONL raises TypeError."""
        file_path = temp_dir / "test.jsonl"
        with pytest.raises(TypeError):
            save_jsonl(sample_data, file_path)


class TestJSONOptions:
    """Tests for JSON options (encoding, formatting, etc.)."""
    
    def test_json_with_unicode(self, temp_dir):
        """Test JSON with Unicode characters."""
        data = {"text": "Hello 世界 🌍", "emoji": "🚀"}
        file_path = temp_dir / "unicode.json"
        
        save_json(data, file_path, ensure_ascii=False)
        loaded_data = load_json(file_path)
        assert loaded_data == data
    
    def test_json_with_custom_indent(self, temp_dir, sample_data):
        """Test JSON with custom indentation."""
        file_path = temp_dir / "indent.json"
        save_json(sample_data, file_path, indent=4)
        
        content = file_path.read_text()
        assert "    " in content  # 4-space indent