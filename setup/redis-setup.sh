#!/bin/bash
# Redis Stack Setup Script for Spark Memory

echo "🚀 Setting up Redis Stack for Spark Memory..."

# 1. Create data directory
REDIS_DATA_DIR="$HOME/.spark-memory/data"
mkdir -p "$REDIS_DATA_DIR"

# 2. Create Redis configuration
cat > "$REDIS_DATA_DIR/redis-stack.conf" << 'EOF'
# Redis Stack Configuration for Memory One Spark

# Data directory
dir ~/.spark-memory/data

# RDB filename
dbfilename dump.rdb

# Save policies (more frequent saves)
save 900 1      # 15 minutes
save 300 10     # 5 minutes  
save 60 10000   # 1 minute

# Disable AOF for simplicity
appendonly no

# Log file
logfile ~/.spark-memory/data/redis-stack.log

# Network
bind 127.0.0.1 ::1
port 6379

# Memory policy
maxmemory-policy allkeys-lru
EOF

# Replace ~ with actual home directory
sed -i '' "s|~|$HOME|g" "$REDIS_DATA_DIR/redis-stack.conf"

# 3. Create LaunchAgent for auto-start
PLIST_FILE="$HOME/Library/LaunchAgents/homebrew.mxcl.redis-stack.plist"
cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>homebrew.mxcl.redis-stack</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/redis-stack-server</string>
        <string>$REDIS_DATA_DIR/redis-stack.conf</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$REDIS_DATA_DIR</string>
    <key>StandardErrorPath</key>
    <string>$REDIS_DATA_DIR/redis-stack.log</string>
    <key>StandardOutPath</key>
    <string>$REDIS_DATA_DIR/redis-stack.log</string>
</dict>
</plist>
EOF

# 4. Load LaunchAgent
launchctl load "$PLIST_FILE"

echo "✅ Redis Stack setup complete!"
echo "📁 Data directory: $REDIS_DATA_DIR"
echo "🔧 Config file: $REDIS_DATA_DIR/redis-stack.conf"
echo "🚀 Redis Stack will start automatically on boot"