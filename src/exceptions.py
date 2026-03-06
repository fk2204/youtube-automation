"""
Custom exception classes for the social distribution system.

Every error type maps to a specific failure mode so callers
can handle them precisely without inspecting error messages.
"""


class SocialDistributionError(Exception):
    """Base class for all social distribution errors."""
    pass


class AuthError(SocialDistributionError):
    """
    Raised when platform authentication fails.

    Triggered on 401 responses, invalid credentials,
    or missing required token types (e.g. Page token vs User token).
    """
    pass


class FileSizeError(SocialDistributionError):
    """
    Raised when a file exceeds the platform's size limit.

    Check the platform's current documentation for exact limits
    as they change frequently.
    """
    pass


class RateLimitError(SocialDistributionError):
    """
    Raised when the platform returns 429 Too Many Requests.

    Includes retry_after seconds when available from the response.
    """

    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(SocialDistributionError):
    """
    Raised when content fails platform-specific validation.

    Examples: carousel with >10 items, missing cover image,
    unsupported video format.
    """
    pass


class UploadError(SocialDistributionError):
    """Raised when a file upload operation fails."""
    pass


class PollingTimeoutError(SocialDistributionError):
    """
    Raised when async media processing polling exceeds max attempts.

    Platforms like Instagram and Pinterest process media asynchronously.
    If status does not reach FINISHED within the attempt limit, this is raised.
    """
    pass


class TierError(SocialDistributionError):
    """
    Raised when the configured API tier does not meet the required minimum.

    Example: TwitterThreadRepurposer requires 'basic' or 'enterprise' tier;
    configuring 'free' raises TierError at __init__ time so the error
    surfaces before any API call is attempted.
    """
    pass


class ConfigError(SocialDistributionError):
    """
    Raised when required configuration is missing or invalid at init time.

    Example: EmailRepurposer raises ConfigError when provider is not in
    SUPPORTED_PROVIDERS, preventing the object from being created with
    an unusable configuration.
    """
    pass
