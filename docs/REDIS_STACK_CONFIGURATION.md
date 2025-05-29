# Redis Stack Configuration Guide for Memory One Spark

## Overview

This guide explains how to properly configure Redis Stack for Memory One Spark, addressing the known issue where `redis-stack-server` ignores the `dir` directive in configuration files.

## The Problem

`redis-stack-server` has a hardcoded default directory (`/opt/homebrew/var/db/redis-stack`) that cannot be overridden through configuration files. This causes issues for users who want to:
- Store Redis data in custom locations
- Use dotfiles for configuration management
- Maintain separate data directories for different projects

## The Solution

Instead of using `redis-stack-server`, we use the standard `redis-server` with manual module loading. This allows full control over data directories and configuration.

## Installation

### Prerequisites

Install Redis Stack via Homebrew:
```bash
brew tap redis-stack/redis-stack
brew install redis-stack
```

### Automatic Setup

Run the provided setup script:
```bash
./setup/redis-setup.sh
```

This script will:
1. Create a custom data directory
2. Generate a proper Redis configuration with AOF enabled
3. Set up a LaunchAgent for automatic startup
4. Use `redis-server` with manual module loading to respect your data directory

### Manual Setup

If you prefer manual configuration:

1. **Find Redis Stack modules location:**
   ```bash
   ls /opt/homebrew/Caskroom/redis-stack-server/
   # Note the version number and use it below
   ```

2. **Start Redis with modules:**
   ```bash
   cd /path/to/your/data/directory
   redis-server redis-stack.conf \
     --loadmodule /opt/homebrew/Caskroom/redis-stack-server/7.4.0-v5/lib/rediscompat.so \
     --loadmodule /opt/homebrew/Caskroom/redis-stack-server/7.4.0-v5/lib/rejson.so \
     --loadmodule /opt/homebrew/Caskroom/redis-stack-server/7.4.0-v5/lib/redisearch.so \
     --loadmodule /opt/homebrew/Caskroom/redis-stack-server/7.4.0-v5/lib/redistimeseries.so \
     --loadmodule /opt/homebrew/Caskroom/redis-stack-server/7.4.0-v5/lib/redisbloom.so
   ```

## Configuration Best Practices

### 1. Enable AOF for Data Durability

```conf
# AOF (Append Only File) persistence
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
appendirname "appendonlydir"
```

### 2. Proper Comment Syntax

Redis configuration files **do not support inline comments**:

```conf
# ❌ WRONG - This will cause errors
save 900 1  # Save after 15 minutes

# ✅ CORRECT - Comments on separate lines
# Save after 15 minutes
save 900 1
```

### 3. Use Absolute Paths

Redis doesn't expand `~` or environment variables:

```conf
# ❌ WRONG
logfile ~/redis.log

# ✅ CORRECT
logfile /Users/username/dotfiles/config/mcp/memory/redis.log
```

## Customizing for Your Environment

### Option 1: Environment Variables (Recommended)

Set these in your shell configuration:
```bash
export REDIS_DATA_DIR="$HOME/your/custom/path"
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Option 2: Direct Configuration

Edit the configuration file directly:
```conf
# Custom data directory (remember: use with redis-server, not redis-stack-server)
dir /your/custom/data/directory
```

## Verifying Your Setup

1. **Check if Redis is running:**
   ```bash
   redis-cli ping
   # Should return: PONG
   ```

2. **Verify modules are loaded:**
   ```bash
   redis-cli MODULE LIST
   # Should show: json, search, timeseries, bloom modules
   ```

3. **Confirm data directory:**
   ```bash
   redis-cli CONFIG GET dir
   # Should show your custom directory
   ```

## Troubleshooting

### Issue: "redis-stack-server ignores my data directory"
**Solution:** Use `redis-server` with manual module loading as shown above.

### Issue: "Configuration file has errors"
**Solution:** Check for inline comments and remove them. All comments must be on separate lines.

### Issue: "Modules not loading"
**Solution:** Verify the module path matches your Redis Stack installation version.

## Platform-Specific Notes

### macOS
- LaunchAgent is recommended for auto-start
- Default module location: `/opt/homebrew/Caskroom/redis-stack-server/VERSION/lib/`

### Linux
- Use systemd service for auto-start
- Module locations vary by distribution

### Windows
- Use Windows Service for auto-start
- WSL2 is recommended for better compatibility

## Security Considerations

1. **Bind to localhost only** in development:
   ```conf
   bind 127.0.0.1 ::1
   ```

2. **Enable password** for production:
   ```conf
   requirepass your-secure-password
   ```

3. **Set proper file permissions**:
   ```bash
   chmod 600 redis-stack.conf
   chmod 700 /path/to/data/directory
   ```

## Additional Resources

- [Redis Configuration Documentation](https://redis.io/docs/management/config/)
- [Redis Persistence Documentation](https://redis.io/docs/manual/persistence/)
- [Redis Stack Modules Documentation](https://redis.io/docs/stack/)