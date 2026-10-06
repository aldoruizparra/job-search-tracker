"""Shared test configuration.

Disables retry backoff so the suite does not spend 15 seconds sleeping to
prove that retries wait. The backoff logic itself is unchanged.
"""

import os

os.environ["BACKOFF_DISABLED"] = "1"
