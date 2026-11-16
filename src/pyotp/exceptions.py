class PyOTPError(Exception):
    """Base class for PyOTP-specific errors."""


class OTPVerificationError(PyOTPError):
    """Raised when an OTP does not match any permitted counter."""


class OTPReplayError(OTPVerificationError):
    """Raised when an OTP matches only at or below the last consumed position.

    Subclass of :class:`OTPVerificationError`, so callers that only handle
    verification failures keep working without changes.
    """
