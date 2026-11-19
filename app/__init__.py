"""HirePulse - AI Job Hunter & Career Agent Package."""

import os
import socket
from dotenv import load_dotenv

# Ensure environment variables are loaded immediately upon package import
load_dotenv()

# Synchronize Langfuse host / base url env vars
if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL", "")
elif os.getenv("LANGFUSE_HOST") and not os.getenv("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_HOST", "")

# Prefer IPv4 over IPv6 when resolving addresses to avoid hanging on networks with unroutable IPv6
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == socket.AF_UNSPEC:
        try:
            return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        except socket.gaierror:
            pass
    return _orig_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4_first_getaddrinfo

__version__ = "0.1.0"
