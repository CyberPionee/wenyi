"""Credential-scoped validation for keys read from the environment."""

import unicodedata


def validate_credential(secret: str | None) -> None:
    """Reject control characters before any HTTP SDK can echo an invalid header.

    A key pasted into an environment file can carry a trailing carriage return or
    newline; the SDK then either fails inside the transport layer with an unrelated
    message or sends a malformed header. Rejecting here names the real problem.
    """
    if secret is not None and any(unicodedata.category(char).startswith("C") for char in secret):
        raise ValueError("API credentials cannot contain control characters; enter the key again")
