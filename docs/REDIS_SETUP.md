# Redis Stack Setup Guide for Spark Memory

## Quick Setup (Recommended)

After installing spark-memory, run:

```bash
spark-memory setup-redis
```

This will automatically:
1. Create data directory at `~/.spark-memory/data`
2. Generate Redis configuration
3. Set up auto-start on boot

## Manual Setup

### 1. Install Redis Stack

**macOS:**
```bash
brew tap redis-stack/redis-stack
brew install redis-stack
```

**Ubuntu/Debian:**
```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack
```

### 2. Configure Redis

Create config file at `~/.spark-memory/data/redis-stack.conf`:

```conf
# Data directory
dir /Users/YOUR_USERNAME/.spark-memory/data

# RDB settings
dbfilename dump.rdb
save 900 1
save 300 10
save 60 10000

# Disable AOF
appendonly no

# Log file
logfile /Users/YOUR_USERNAME/.spark-memory/data/redis-stack.log

# Network
bind 127.0.0.1 ::1
port 6379
```

### 3. Set up Auto-start

**macOS (LaunchAgent):**

Create `~/Library/LaunchAgents/homebrew.mxcl.redis-stack.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>homebrew.mxcl.redis-stack</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/redis-stack-server</string>
        <string>/Users/YOUR_USERNAME/.spark-memory/data/redis-stack.conf</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/.spark-memory/data</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/.spark-memory/data/redis-stack.log</string>
    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/.spark-memory/data/redis-stack.log</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/homebrew.mxcl.redis-stack.plist
```

**Linux (systemd):**

Create `/etc/systemd/system/redis-stack.service`:

```ini
[Unit]
Description=Redis Stack
After=network.target

[Service]
Type=notify
ExecStart=/usr/bin/redis-stack-server /home/YOUR_USERNAME/.spark-memory/data/redis-stack.conf
TimeoutStopSec=0
Restart=always
User=YOUR_USERNAME
Group=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/.spark-memory/data

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable redis-stack
sudo systemctl start redis-stack
```

## Verify Installation

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Check data directory
redis-cli CONFIG GET dir
# Should return: /Users/YOUR_USERNAME/.spark-memory/data

# Check loaded modules
redis-cli MODULE LIST
# Should show: search, timeseries, ReJSON, bf, etc.
```

## Troubleshooting

### Port already in use
```bash
# Find process using port 6379
lsof -i :6379

# Kill existing Redis
redis-cli shutdown
```

### Permission denied
```bash
# Fix directory permissions
chmod 755 ~/.spark-memory/data
```

### Data not persisting
- Check if `appendonly` is set to `no` in config
- Verify `save` policies are configured
- Check log file for errors

## Data Migration

If you have existing data to migrate:

```bash
# Copy existing dump.rdb to new location
cp /path/to/old/dump.rdb ~/.spark-memory/data/

# Restart Redis
redis-cli shutdown
# Redis will restart automatically via LaunchAgent/systemd
```