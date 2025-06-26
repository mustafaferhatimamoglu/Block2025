import time
from typing import Optional, Dict, Any

import requests


def get_with_retry(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    max_retries: int = 5,
) -> requests.Response:
    """Perform GET request with retries on network errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    # Should never reach here
    raise RuntimeError("Unreachable")
