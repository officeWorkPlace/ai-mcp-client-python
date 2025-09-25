"""
Rate limiting utilities
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque


class RateLimiter:
    """Simple rate limiter implementation"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()

    def allow_request(self, identifier: str = "default") -> bool:
        """
        Check if a request is allowed

        Args:
            identifier: Request identifier (e.g., user ID, IP)

        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()

        # Remove old requests outside the window
        while self.requests and current_time - self.requests[0] > self.window_seconds:
            self.requests.popleft()

        # Check if we're under the limit
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True

        return False

    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window"""
        current_time = time.time()

        # Remove old requests
        while self.requests and current_time - self.requests[0] > self.window_seconds:
            self.requests.popleft()

        return max(0, self.max_requests - len(self.requests))

    def get_reset_time(self) -> Optional[float]:
        """Get timestamp when rate limit will reset"""
        if not self.requests:
            return None

        return self.requests[0] + self.window_seconds


class TokenBucketRateLimiter:
    """Token bucket rate limiter for more sophisticated rate limiting"""

    def __init__(self, capacity: int = 100, refill_rate: float = 1.0):
        """
        Initialize token bucket rate limiter

        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Rate at which tokens are added (tokens per second)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def allow_request(self, tokens_required: int = 1) -> bool:
        """
        Check if a request is allowed and consume tokens

        Args:
            tokens_required: Number of tokens required for this request

        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()

        # Refill tokens based on elapsed time
        elapsed = current_time - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = current_time

        # Check if we have enough tokens
        if self.tokens >= tokens_required:
            self.tokens -= tokens_required
            return True

        return False

    def get_available_tokens(self) -> float:
        """Get number of available tokens"""
        current_time = time.time()
        elapsed = current_time - self.last_refill
        return min(self.capacity, self.tokens + elapsed * self.refill_rate)


class MultiUserRateLimiter:
    """Rate limiter that handles multiple users/identifiers"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize multi-user rate limiter

        Args:
            max_requests: Maximum requests per user in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_requests: Dict[str, deque] = defaultdict(deque)

    def allow_request(self, identifier: str) -> bool:
        """
        Check if a request is allowed for a specific user

        Args:
            identifier: User identifier

        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()
        user_queue = self.user_requests[identifier]

        # Remove old requests outside the window
        while user_queue and current_time - user_queue[0] > self.window_seconds:
            user_queue.popleft()

        # Check if we're under the limit
        if len(user_queue) < self.max_requests:
            user_queue.append(current_time)
            return True

        return False

    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests for a user"""
        current_time = time.time()
        user_queue = self.user_requests[identifier]

        # Remove old requests
        while user_queue and current_time - user_queue[0] > self.window_seconds:
            user_queue.popleft()

        return max(0, self.max_requests - len(user_queue))

    def cleanup_old_users(self, inactive_threshold: int = 86400) -> None:
        """
        Clean up data for users who haven't made requests recently

        Args:
            inactive_threshold: Seconds of inactivity after which to remove user data
        """
        current_time = time.time()
        inactive_users = []

        for user_id, user_queue in self.user_requests.items():
            if not user_queue or current_time - user_queue[-1] > inactive_threshold:
                inactive_users.append(user_id)

        for user_id in inactive_users:
            del self.user_requests[user_id]
