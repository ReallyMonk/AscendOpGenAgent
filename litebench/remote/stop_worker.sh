#!/bin/bash
# Stop OpenOps Remote Worker Server

echo "Stopping OpenOps Worker Server..."

# Find the worker_server.py process
PID=$(ps aux | grep "worker_server.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "No worker server is running."
    exit 0
fi

echo "Found worker server process: $PID"
echo "Sending SIGTERM..."
kill $PID

# Wait for graceful shutdown
sleep 2

# Check if still running
if ps -p $PID > /dev/null 2>&1; then
    echo "Process still running, sending SIGKILL..."
    kill -9 $PID
    sleep 1
fi

# Verify stopped
if ps -p $PID > /dev/null 2>&1; then
    echo "❌ Failed to stop worker server"
    exit 1
else
    echo "✅ Worker server stopped successfully"
    exit 0
fi
