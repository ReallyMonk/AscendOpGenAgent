#!/bin/bash
# Start OpenOps Remote Worker Server

# Default values
PORT=9001
HOST="0.0.0.0"
LOG_DIR=""
MAX_WORKERS=4
DEVICES="0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --log-dir LOG_DIR [--port PORT] [--host HOST] [--max-workers MAX_WORKERS] [--devices DEVICES]"
            echo ""
            echo "Options:"
            echo "  --port PORT           Server port (default: 9001)"
            echo "  --host HOST           Server host (default: 0.0.0.0)"
            echo "  --log-dir LOG_DIR     Directory for storing logs (required)"
            echo "  --max-workers N       Maximum concurrent workers (default: 4)"
            echo "  --devices DEVICES     Comma-separated NPU device IDs (default: 0)"
            echo ""
            echo "Example:"
            echo "  $0 --log-dir /server/logs --port 9001 --devices 0,1,2,3 --max-workers 4"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$LOG_DIR" ]; then
    echo "Error: --log-dir is required"
    echo "Usage: $0 --log-dir LOG_DIR [--port PORT] [--host HOST] [--max-workers MAX_WORKERS] [--devices DEVICES]"
    exit 1
fi

# Check CANN environment
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "Error: ASCEND_HOME_PATH environment variable is not set"
    echo "Please set it to your CANN installation path, e.g.:"
    echo "  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest"
    exit 1
fi

if [ ! -d "$ASCEND_HOME_PATH" ]; then
    echo "Error: ASCEND_HOME_PATH directory does not exist: $ASCEND_HOME_PATH"
    exit 1
fi

# Source CANN environment for LD_LIBRARY_PATH etc.
SETENV_SCRIPT="$ASCEND_HOME_PATH/bin/setenv.bash"
if [ -f "$SETENV_SCRIPT" ]; then
    source "$SETENV_SCRIPT"
fi

# Add Ascend driver libraries (required for aclInit / torch.npu)
DRIVER_LIB="/usr/local/Ascend/driver/lib64"
if [ -d "$DRIVER_LIB" ]; then
    export LD_LIBRARY_PATH="$DRIVER_LIB:$DRIVER_LIB/driver:${LD_LIBRARY_PATH}"
fi

echo "========================================="
echo "OpenOps Remote Worker Server"
echo "========================================="
echo "CANN Path: $ASCEND_HOME_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Directory: $LOG_DIR"
echo "Max Workers: $MAX_WORKERS"
echo "NPU Devices: $DEVICES"
echo "========================================="

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure gen_project.py is available in server directory for standalone deployment
GEN_PROJECT_SRC="$SCRIPT_DIR/../.opencode/skills/ascend_call_generation/scripts/gen_project.py"
GEN_PROJECT_DST="$SCRIPT_DIR/gen_project.py"
if [ ! -f "$GEN_PROJECT_DST" ] && [ -f "$GEN_PROJECT_SRC" ]; then
    echo "Copying gen_project.py to server directory..."
    cp "$GEN_PROJECT_SRC" "$GEN_PROJECT_DST"
fi

# Start server
python3 "$SCRIPT_DIR/worker_server.py" \
    --host "$HOST" \
    --port "$PORT" \
    --log-dir "$LOG_DIR" \
    --max-workers "$MAX_WORKERS" \
    --devices "$DEVICES"
