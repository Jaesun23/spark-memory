#!/bin/bash
# Redis Stack Setup Script for Spark Memory

set -e  # Exit on error

echo "🚀 Setting up Redis Stack for Spark Memory..."

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

echo "📦 Platform detected: $PLATFORM"

# Check if Redis Stack is installed
check_redis_stack() {
    if command -v redis-stack-server &> /dev/null; then
        echo "✅ Redis Stack is already installed"
        return 0
    else
        echo "❌ Redis Stack is not installed"
        return 1
    fi
}

# Install Redis Stack
install_redis_stack() {
    echo "📥 Installing Redis Stack..."
    
    if [[ "$PLATFORM" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "❌ Homebrew is required but not installed"
            echo "Please install Homebrew from https://brew.sh"
            exit 1
        fi
        
        echo "🍺 Installing Redis Stack using Homebrew..."
        brew tap redis-stack/redis-stack
        brew install redis-stack
        
    elif [[ "$PLATFORM" == "linux" ]]; then
        # Detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
            VER=$VERSION_ID
        else
            echo "❌ Cannot detect Linux distribution"
            exit 1
        fi
        
        if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
            echo "🐧 Installing Redis Stack on Ubuntu/Debian..."
            curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
            sudo apt-get update
            sudo apt-get install -y redis-stack-server
            
        elif [[ "$OS" == "fedora" ]] || [[ "$OS" == "centos" ]] || [[ "$OS" == "rhel" ]]; then
            echo "🐧 Installing Redis Stack on Fedora/CentOS/RHEL..."
            sudo dnf install -y redis-stack-server
            
        else
            echo "❌ Unsupported Linux distribution: $OS"
            echo "Please install Redis Stack manually from https://redis.io/docs/install/install-stack/"
            exit 1
        fi
    fi
}

# Check and install Redis Stack if needed
if ! check_redis_stack; then
    read -p "Would you like to install Redis Stack now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_redis_stack
        
        # Verify installation
        if ! check_redis_stack; then
            echo "❌ Redis Stack installation failed"
            exit 1
        fi
    else
        echo "❌ Redis Stack is required for Spark Memory"
        echo "Please install it manually from https://redis.io/docs/install/install-stack/"
        exit 1
    fi
fi

# Verify Redis Stack modules
echo "🔍 Verifying Redis Stack modules..."
REQUIRED_MODULES=("ReJSON" "search" "timeseries")
MISSING_MODULES=()

# Start Redis Stack temporarily to check modules
redis-stack-server --daemonize yes --port 16379 > /dev/null 2>&1
sleep 2

for module in "${REQUIRED_MODULES[@]}"; do
    if ! redis-cli -p 16379 MODULE LIST 2>/dev/null | grep -qi "$module"; then
        MISSING_MODULES+=("$module")
    fi
done

# Stop temporary Redis instance
redis-cli -p 16379 SHUTDOWN > /dev/null 2>&1

if [ ${#MISSING_MODULES[@]} -ne 0 ]; then
    echo "❌ Missing required Redis modules: ${MISSING_MODULES[*]}"
    echo "Please ensure you have Redis Stack (not just Redis) installed"
    exit 1
fi

echo "✅ All required Redis modules are available"

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
if [[ "$PLATFORM" == "macos" ]]; then
    sed -i '' "s|~|$HOME|g" "$REDIS_DATA_DIR/redis-stack.conf"
else
    sed -i "s|~|$HOME|g" "$REDIS_DATA_DIR/redis-stack.conf"
fi

# 3. Setup auto-start based on platform
if [[ "$PLATFORM" == "macos" ]]; then
    # Create LaunchAgent for macOS
    mkdir -p "$HOME/Library/LaunchAgents"
    PLIST_FILE="$HOME/Library/LaunchAgents/homebrew.mxcl.redis-stack.plist"
    
    # Find Redis Stack binary location
    REDIS_STACK_BIN=$(which redis-stack-server)
    
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>homebrew.mxcl.redis-stack</string>
    <key>ProgramArguments</key>
    <array>
        <string>$REDIS_STACK_BIN</string>
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
    
    # Load LaunchAgent
    launchctl load "$PLIST_FILE" 2>/dev/null || true
    
    echo "✅ Redis Stack LaunchAgent created at: $PLIST_FILE"
    echo "🚀 Redis Stack will start automatically on boot"
    
elif [[ "$PLATFORM" == "linux" ]]; then
    # Create systemd service for Linux
    SERVICE_FILE="$HOME/.config/systemd/user/redis-stack.service"
    mkdir -p "$HOME/.config/systemd/user"
    
    # Find Redis Stack binary location
    REDIS_STACK_BIN=$(which redis-stack-server)
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Redis Stack server for Spark Memory
After=network.target

[Service]
Type=notify
ExecStart=$REDIS_STACK_BIN $REDIS_DATA_DIR/redis-stack.conf
WorkingDirectory=$REDIS_DATA_DIR
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
EOF
    
    # Reload systemd and enable service
    systemctl --user daemon-reload
    systemctl --user enable redis-stack.service
    systemctl --user start redis-stack.service
    
    echo "✅ Redis Stack systemd service created at: $SERVICE_FILE"
    echo "🚀 Redis Stack will start automatically on boot"
fi

# 4. Start Redis Stack
echo "🚀 Starting Redis Stack..."
if [[ "$PLATFORM" == "macos" ]]; then
    launchctl start homebrew.mxcl.redis-stack
elif [[ "$PLATFORM" == "linux" ]]; then
    systemctl --user start redis-stack.service
fi

# Wait for Redis Stack to start
sleep 2

# 5. Verify Redis Stack is running
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis Stack is running!"
else
    echo "⚠️  Redis Stack may not have started properly"
    echo "Check logs at: $REDIS_DATA_DIR/redis-stack.log"
fi

echo ""
echo "✅ Redis Stack setup complete!"
echo "📁 Data directory: $REDIS_DATA_DIR"
echo "🔧 Config file: $REDIS_DATA_DIR/redis-stack.conf"
echo "📋 Logs: $REDIS_DATA_DIR/redis-stack.log"
echo ""
echo "🎯 Quick commands:"
if [[ "$PLATFORM" == "macos" ]]; then
    echo "  Start:   launchctl start homebrew.mxcl.redis-stack"
    echo "  Stop:    launchctl stop homebrew.mxcl.redis-stack"
    echo "  Status:  launchctl list | grep redis-stack"
elif [[ "$PLATFORM" == "linux" ]]; then
    echo "  Start:   systemctl --user start redis-stack"
    echo "  Stop:    systemctl --user stop redis-stack"
    echo "  Status:  systemctl --user status redis-stack"
fi
echo "  Test:    redis-cli ping"
echo "  Modules: redis-cli MODULE LIST"