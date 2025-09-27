"""
Cross-platform tests for utility functions
"""

import pytest
import tempfile
import json
import platform
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_client.utils.validators import InputValidator

try:
    from mcp_client.utils.rate_limiter import RateLimiter, TokenBucketRateLimiter
    from mcp_client.utils.helpers import (
        load_json_file, save_json_file, format_duration, format_bytes,
        safe_get_nested, merge_dicts, validate_environment
    )
    HAS_HELPERS = True
except ImportError:
    # Some utilities might not exist in new structure yet
    HAS_HELPERS = False


class TestInputValidator:
    """Test input validation"""
    
    def setup_method(self):
        self.validator = InputValidator()
    
    def test_valid_queries(self):
        """Test validation of valid queries"""
        valid_queries = [
            "What is the weather today?",
            "Calculate 2 + 2",
            "List files in current directory",
            "Show me information about Python",
        ]
        
        for query in valid_queries:
            assert self.validator.validate_query(query) is True
    
    def test_invalid_queries(self):
        """Test validation of invalid queries"""
        invalid_queries = [
            "",  # Empty string
            None,  # None value
            "rm -rf /",  # Dangerous command
            "DROP TABLE users",  # Dangerous SQL
            "<script>alert('xss')</script>",  # XSS attempt
            "javascript:void(0)",  # JavaScript protocol
            "eval('malicious code')",  # Code execution
            "a" * 20000,  # Too long
            "@@@@####%%%%$$$$" * 100,  # Too many special chars
        ]
        
        for query in invalid_queries:
            assert self.validator.validate_query(query) is False
    
    def test_url_validation(self):
        """Test URL validation"""
        valid_urls = [
            "https://example.com",
            "http://localhost:8080",
            "https://api.example.com/v1/data",
        ]
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Not http/https
            "",
            None,
        ]
        
        for url in valid_urls:
            assert self.validator.validate_url(url) is True
        
        for url in invalid_urls:
            assert self.validator.validate_url(url) is False
    
    def test_file_path_validation_cross_platform(self, platform_paths):
        """Test cross-platform file path validation"""
        # Test platform-specific safe paths
        for path in platform_paths["safe_paths"]:
            assert self.validator.validate_file_path(path) is True, f"Safe path should be valid: {path}"

        # Test platform-specific dangerous paths
        for path in platform_paths["dangerous_paths"]:
            assert self.validator.validate_file_path(path) is False, f"Dangerous path should be blocked: {path}"

        # Test universal invalid paths
        universal_invalid = ["", None, ".."]
        for path in universal_invalid:
            assert self.validator.validate_file_path(path) is False

    def test_file_path_validation_platform_specific(self):
        """Test platform-specific path validation logic"""
        current_platform = platform.system()

        if current_platform == "Windows":
            # Windows-specific tests
            windows_dangerous = [
                "C:\\Windows\\System32\\",
                "C:\\Program Files\\",
                "C:\\ProgramData\\",
                "\\Windows\\System32\\"
            ]
            for path in windows_dangerous:
                assert self.validator.validate_file_path(path) is False, f"Windows path should be blocked: {path}"

        elif current_platform in ["Linux", "Darwin"]:
            # Unix-like systems tests
            unix_dangerous = [
                "/etc/passwd",
                "/bin/sh",
                "/usr/bin/sudo",
                "/System/Library/",  # macOS
                "/private/etc/"       # macOS
            ]
            for path in unix_dangerous:
                assert self.validator.validate_file_path(path) is False, f"Unix path should be blocked: {path}"
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        test_cases = [
            ("normal text", "normal text"),
            ("text\x00with\x01control\x02chars", "textwithcontrolchars"),
            ("text\nwith\twhitespace", "text\nwith\twhitespace"),  # Keep newlines and tabs
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            result = self.validator.sanitize_input(input_text)
            assert result == expected


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # First 3 requests should be allowed
        for _ in range(3):
            assert limiter.allow_request() is True
        
        # 4th request should be denied
        assert limiter.allow_request() is False
    
    def test_token_bucket_rate_limiter(self):
        """Test token bucket rate limiter"""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
        
        # Should allow initial requests up to capacity
        for _ in range(5):
            assert limiter.allow_request() is True
        
        # Should deny additional request
        assert limiter.allow_request() is False
        
        # Check available tokens
        assert limiter.get_available_tokens() < 1.0
    
    def test_rate_limiter_remaining_requests(self):
        """Test getting remaining requests"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Use 2 requests
        limiter.allow_request()
        limiter.allow_request()
        
        # Should have 3 remaining
        assert limiter.get_remaining_requests() == 3


class TestHelpers:
    """Test helper functions"""
    
    def test_json_file_operations(self, tmp_path):
        """Test JSON file loading and saving"""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = tmp_path / "test.json"
        
        # Save JSON
        save_json_file(test_data, file_path)
        assert file_path.exists()
        
        # Load JSON
        loaded_data = load_json_file(file_path)
        assert loaded_data == test_data
    
    def test_format_duration(self):
        """Test duration formatting"""
        test_cases = [
            (30, "30.0s"),
            (90, "1.5m"),
            (3661, "1.0h"),
        ]
        
        for seconds, expected in test_cases:
            assert format_duration(seconds) == expected
    
    def test_format_bytes(self):
        """Test byte formatting"""
        test_cases = [
            (512, "512.0B"),
            (1536, "1.5KB"),
            (1048576, "1.0MB"),
            (1073741824, "1.0GB"),
        ]
        
        for bytes_count, expected in test_cases:
            assert format_bytes(bytes_count) == expected
    
    def test_safe_get_nested(self):
        """Test safe nested dictionary access"""
        data = {
            "level1": {
                "level2": {
                    "value": "found"
                }
            }
        }
        
        # Existing path
        result = safe_get_nested(data, ["level1", "level2", "value"])
        assert result == "found"
        
        # Non-existing path
        result = safe_get_nested(data, ["level1", "missing", "value"], default="default")
        assert result == "default"
        
        # Empty keys
        result = safe_get_nested(data, [], default="root")
        assert result == data
    
    def test_merge_dicts(self):
        """Test dictionary merging"""
        dict1 = {
            "a": 1,
            "b": {"nested": "value1"},
            "c": "unchanged"
        }
        
        dict2 = {
            "a": 2,  # Override
            "b": {"added": "value2"},  # Merge nested
            "d": "new"  # Add new
        }
        
        result = merge_dicts(dict1, dict2)
        
        assert result["a"] == 2  # Overridden
        assert result["b"]["nested"] == "value1"  # Preserved
        assert result["b"]["added"] == "value2"  # Added
        assert result["c"] == "unchanged"  # Unchanged
        assert result["d"] == "new"  # New key
    
    def test_validate_environment(self):
        """Test environment validation"""
        issues = validate_environment()
        
        # Should return a list (empty if environment is valid)
        assert isinstance(issues, list)
        
        # In test environment, might have missing packages
        # Just ensure function doesn't crash


@pytest.mark.asyncio
class TestAsyncHelpers:
    """Test async helper functions"""
    
    async def test_async_operations(self):
        """Test that helper functions work in async context"""
        # Test that helpers can be used in async functions
        data = {"test": "data"}
        
        # This should work without blocking the event loop
        result = safe_get_nested(data, ["test"])
        assert result == "data"
