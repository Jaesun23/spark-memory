# Redis Setup Migration Guide

If you already have `spark-memory` installed and are experiencing issues with Redis data directory, follow this guide.

## The Issue

`redis-stack-server` ignores the `dir` directive in configuration files, always using the default directory:
- macOS: `/opt/homebrew/var/db/redis-stack`
- Linux: `/var/lib/redis-stack`

## Quick Fix

Run the updated setup script:
```bash
./setup/redis-setup.sh
```

This will:
1. Stop any existing Redis Stack instance
2. Create a new configuration that uses `redis-server` with manual module loading
3. Set up your custom data directory properly
4. Restart Redis with the correct settings

## What Changed?

### Old Method (Doesn't work for custom directories):
```bash
redis-stack-server /path/to/config
```

### New Method (Works correctly):
```bash
redis-server /path/to/config \
  --loadmodule /path/to/rejson.so \
  --loadmodule /path/to/redisearch.so \
  --loadmodule /path/to/redistimeseries.so \
  --loadmodule /path/to/redisbloom.so
```

## Verify the Fix

Check your data directory:
```bash
redis-cli CONFIG GET dir
```

Should show your custom directory, not the default one.

## No Data Loss

Your existing Redis data is safe. The new setup will use the same data directory you configured.