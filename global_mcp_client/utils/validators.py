"""
Input validation utilities
"""

import re
from typing import List, Optional
from urllib.parse import urlparse


class InputValidator:
    """Validates user inputs for security and correctness"""

    def __init__(self):
        # Dangerous patterns to block
        self.dangerous_patterns = [
            r"(?i)\b(rm\s+-rf|del\s+/[qsf]|format\s+c:)",  # Dangerous system commands
            r"(?i)\b(drop\s+table|delete\s+from.*where\s+1=1)",  # Dangerous SQL
            r"<script[^>]*>.*?</script>",  # Script tags
            r"javascript:",  # JavaScript protocol
            r"(?i)\b(eval|exec)\s*\(",  # Code execution functions
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern) for pattern in self.dangerous_patterns
        ]

    def validate_query(self, query: str) -> bool:
        """
        Validate a user query for safety

        Args:
            query: User input query

        Returns:
            True if query is safe, False otherwise
        """
        if not query or not isinstance(query, str):
            return False

        # Check length
        if len(query) > 10000:  # Reasonable limit
            return False

        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(query):
                return False

        # Check for excessive special characters (potential injection)
        special_char_ratio = sum(
            1 for c in query if not c.isalnum() and not c.isspace()
        ) / len(query)
        if special_char_ratio > 0.5:
            return False

        return True

    def validate_url(self, url: str) -> bool:
        """
        Validate a URL (only HTTP/HTTPS)

        Args:
            url: URL to validate

        Returns:
            True if URL is valid HTTP/HTTPS, False otherwise
        """
        try:
            result = urlparse(url)
            return result.scheme in ["http", "https"] and bool(result.netloc)
        except:
            return False

    def validate_file_path(self, path: str) -> bool:
        """
        Validate a file path for safety

        Args:
            path: File path to validate

        Returns:
            True if path is safe, False otherwise
        """
        if not path or not isinstance(path, str):
            return False

        # Block dangerous paths
        dangerous_paths = [
            "..",
            "/etc/",
            "/bin/",
            "/usr/bin/",
            "C:\\Windows\\",
            "C:\\Program Files\\",
        ]

        for dangerous in dangerous_paths:
            if dangerous in path:
                return False

        return True

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Remove control characters except newline and tab
        sanitized = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        # Limit length
        return sanitized[:10000]

    def validate_json(self, json_str: str) -> bool:
        """
        Validate JSON string

        Args:
            json_str: JSON string to validate

        Returns:
            True if valid JSON, False otherwise
        """
        try:
            import json

            json.loads(json_str)
            return True
        except:
            return False
