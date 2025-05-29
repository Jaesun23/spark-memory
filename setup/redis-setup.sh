#!/bin/bash
# Redis Stack Setup Script for Spark Memory
# 
# IMPORTANT: redis-stack-server has a known issue where it ignores the 'dir' 
# directive in config files. We must pass --dir as a command-line argument.

echo "🚀 Setting up Redis Stack for Spark Memory..."

# 1. Create data directory
REDIS_DATA_DIR="$HOME/dotfiles/config/mcp/memory"
mkdir -p "$REDIS_DATA_DIR"

# 2. Create Redis configuration
cat > "$REDIS_DATA_DIR/redis-stack.conf" << EOF
# Redis Stack Configuration for Memory One Spark
# Note: Comments must be on their own lines - inline comments are not supported

# Network binding
bind 127.0.0.1 ::1
port 6379

# Data persistence - RDB snapshots
# Save the DB if both conditions are met:
# after X seconds if at least Y keys changed
save 900 1
save 300 10
save 60 10000

# RDB filename
dbfilename dump.rdb

# AOF (Append Only File) persistence
# Recommended for better durability
appendonly yes
appendfilename "appendonly.aof"

# AOF directory name
appendirname "appendonlydir"

# fsync policy:
# no: let the OS flush data when it wants (fastest, less safe)
# always: fsync after every write (slowest, safest)
# everysec: fsync every second (good compromise)
appendfsync everysec

# Prevent fsync during rewrites
no-appendfsync-on-rewrite no

# Auto rewrite the AOF when it gets too big
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Use RDB format for AOF preamble for faster loading
aof-use-rdb-preamble yes

# Log file (absolute path required)
logfile $REDIS_DATA_DIR/redis-stack.log

# Log level
loglevel notice

# Memory management
maxmemory-policy allkeys-lru

# Disable protected mode for local development
protected-mode no
EOF

# 3. Find Redis Stack modules location
REDIS_STACK_DIR="/opt/homebrew/Caskroom/redis-stack-server"
if [ ! -d "$REDIS_STACK_DIR" ]; then
    echo "❌ Redis Stack not found. Please install with: brew install redis-stack"
    exit 1
fi

# Get the latest version directory
REDIS_STACK_VERSION=$(ls -1 "$REDIS_STACK_DIR" | sort -V | tail -1)
MODULES_DIR="$REDIS_STACK_DIR/$REDIS_STACK_VERSION/lib"

echo "📦 Found Redis Stack modules at: $MODULES_DIR"

# 4. Create LaunchAgent for auto-start
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
        <string>/opt/homebrew/bin/redis-server</string>
        <string>$REDIS_DATA_DIR/redis-stack.conf</string>
        <string>--loadmodule</string>
        <string>$MODULES_DIR/rediscompat.so</string>
        <string>--loadmodule</string>
        <string>$MODULES_DIR/rejson.so</string>
        <string>--loadmodule</string>
        <string>$MODULES_DIR/redisearch.so</string>
        <string>--loadmodule</string>
        <string>$MODULES_DIR/redistimeseries.so</string>
        <string>--loadmodule</string>
        <string>$MODULES_DIR/redisbloom.so</string>
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

# 5. Stop existing Redis Stack if running
echo "🛑 Stopping any existing Redis Stack service..."
launchctl unload "$PLIST_FILE" 2>/dev/null || true

# Also stop any redis-stack-server processes
pkill -f redis-stack-server 2>/dev/null || true

# 6. Load LaunchAgent
echo "🚀 Starting Redis Stack service with proper data directory..."
launchctl load "$PLIST_FILE"

# 7. Wait for Redis to start
echo "⏳ Waiting for Redis to start..."
sleep 3

# 8. Test connection and verify modules
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis Stack is running and responding to commands"
    
    # Check if modules are loaded
    echo "🔍 Checking loaded modules..."
    redis-cli MODULE LIST | grep -E "(json|search|timeseries|bloom)" > /dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Redis Stack modules loaded successfully"
    else
        echo "⚠️  Some Redis Stack modules may not be loaded. Check logs for details."
    fi
else
    echo "❌ Redis Stack failed to start. Check logs at: $REDIS_DATA_DIR/redis-stack.log"
    exit 1
fi

echo ""
echo "✅ Redis Stack setup complete!"
echo "📁 Data directory: $REDIS_DATA_DIR"
echo "🔧 Config file: $REDIS_DATA_DIR/redis-stack.conf"
echo "📝 Log file: $REDIS_DATA_DIR/redis-stack.log"
echo "🚀 Redis Stack will start automatically on boot"
echo ""
echo "⚠️  Note: Using redis-server with manual module loading to respect custom data directory"
echo "⚠️  This works around the redis-stack-server issue that ignores the 'dir' directive"
echo ""
echo "Useful commands:"
echo "  Start:   launchctl load $PLIST_FILE"
echo "  Stop:    launchctl unload $PLIST_FILE"
echo "  Status:  redis-cli ping"
echo "  Modules: redis-cli MODULE LIST"
echo "  Logs:    tail -f $REDIS_DATA_DIR/redis-stack.log"
