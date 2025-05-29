#!/usr/bin/env python3
"""Setup script for Redis Stack configuration."""

import os
import subprocess
import sys


def setup_redis():
    """Run the Redis setup script."""
    setup_script = os.path.join(
        os.path.dirname(__file__),
        "..",
        "setup",
        "redis-setup.sh"
    )

    if not os.path.exists(setup_script):
        print("❌ Setup script not found!")
        sys.exit(1)

    # Make it executable
    os.chmod(setup_script, 0o755)

    # Run the script
    subprocess.run(["/bin/bash", setup_script])


if __name__ == "__main__":
    setup_redis()