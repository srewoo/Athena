#!/bin/bash

# PromptCritic - Start Script
# This script starts both the backend and frontend servers

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   PromptCritic - Starting Servers${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    # Kill all child processes
    pkill -P $$ 2>/dev/null || true
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Function to kill process on a specific port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Port $port is in use by process $pid. Killing it...${NC}"
        kill -9 $pid 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}✓ Port $port is now free${NC}"
    fi
}

# Function to clear caches and logs
clear_caches() {
    echo -e "${BLUE}Clearing caches and logs...${NC}"
    
    # Clear backend caches
    if [ -d "$BACKEND_DIR" ]; then
        echo -e "${YELLOW}  - Clearing backend Python cache...${NC}"
        find "$BACKEND_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "$BACKEND_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
        find "$BACKEND_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
        
        # Clear backend log
        if [ -f "$BACKEND_DIR/server.log" ]; then
            rm -f "$BACKEND_DIR/server.log"
            echo -e "${YELLOW}  - Cleared backend server.log${NC}"
        fi
        
        echo -e "${GREEN}  ✓ Backend cache cleared${NC}"
    fi
    
    # Clear frontend caches
    if [ -d "$FRONTEND_DIR" ]; then
        echo -e "${YELLOW}  - Clearing frontend cache...${NC}"
        
        # Clear node_modules cache
        if [ -d "$FRONTEND_DIR/node_modules/.cache" ]; then
            rm -rf "$FRONTEND_DIR/node_modules/.cache"
            echo -e "${YELLOW}  - Cleared node_modules/.cache${NC}"
        fi
        
        # Clear build directory
        if [ -d "$FRONTEND_DIR/build" ]; then
            rm -rf "$FRONTEND_DIR/build"
            echo -e "${YELLOW}  - Cleared build directory${NC}"
        fi
        
        # Clear yarn cache (optional, only if .yarn/cache exists)
        if [ -d "$FRONTEND_DIR/.yarn/cache" ]; then
            rm -rf "$FRONTEND_DIR/.yarn/cache"
            echo -e "${YELLOW}  - Cleared .yarn/cache${NC}"
        fi
        
        echo -e "${GREEN}  ✓ Frontend cache cleared${NC}"
    fi
    
    echo -e "${GREEN}✓ All caches and logs cleared!${NC}\n"
}

# Check if backend directory exists
if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${RED}Error: Backend directory not found at $BACKEND_DIR${NC}"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    echo -e "${RED}Error: Frontend directory not found at $FRONTEND_DIR${NC}"
    exit 1
fi

# Clear all caches and logs
clear_caches

# Kill any processes on ports 8000 and 3000
echo -e "${BLUE}Checking for processes on ports 8000 and 3000...${NC}"
kill_port 8000
kill_port 3000
echo ""

# Start Backend Server
echo -e "${GREEN}[1/2] Starting Backend Server...${NC}"
echo -e "Directory: $BACKEND_DIR"

# Check if virtual environment exists
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not found. Creating one...${NC}"
    cd "$BACKEND_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd "$SCRIPT_DIR"
fi

# Check if .env exists
if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in backend directory${NC}"
    echo -e "${YELLOW}Creating a default .env from .env.example if it exists...${NC}"
    if [ -f "$BACKEND_DIR/env.example" ]; then
        cp "$BACKEND_DIR/env.example" "$BACKEND_DIR/.env"
        echo -e "${GREEN}Created .env from env.example${NC}"
    elif [ -f "$SCRIPT_DIR/env.example" ]; then
        cp "$SCRIPT_DIR/env.example" "$BACKEND_DIR/.env"
        echo -e "${GREEN}Created .env from root env.example${NC}"
    else
        echo -e "${RED}No .env.example found. Please configure .env manually.${NC}"
    fi
fi

# Start backend in background with logging
BE_LOG="$BACKEND_DIR/server.log"
(
    cd "$BACKEND_DIR"
    source venv/bin/activate
    echo -e "${GREEN}✓ Backend virtual environment activated${NC}"
    echo -e "${GREEN}✓ Starting FastAPI server on http://localhost:8000${NC}"
    # Use tee to show output and save to log
    uvicorn server:app --reload --host 0.0.0.0 --port 8000 > "$BE_LOG" 2>&1
) &

BACKEND_PID=$!
echo -e "${GREEN}✓ Backend server process spawned (PID: $BACKEND_PID)${NC}"

# Health check
echo -e "${BLUE}Waiting for backend to become healthy...${NC}"
MAX_RETRIES=30
COUNT=0
HEALTHY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/docs > /dev/null; then
        HEALTHY=1
        break
    fi
    # Check if process is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}Backend process died unexpectedly!${NC}"
        echo -e "${RED}Last 10 lines of log ($BE_LOG):${NC}"
        tail -n 10 "$BE_LOG"
        exit 1
    fi
    sleep 1
    echo -n "."
    COUNT=$((COUNT+1))
done
echo ""

if [ $HEALTHY -eq 1 ]; then
    echo -e "${GREEN}✓ Backend is healthy and responding!${NC}\n"
else
    echo -e "${RED}Backend failed to start within $MAX_RETRIES seconds.${NC}"
    echo -e "${RED}Check logs at $BE_LOG${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    echo -e "${RED}Last 10 lines of log:${NC}"
    tail -n 10 "$BE_LOG"
    exit 1
fi

# Start Frontend Server
echo -e "${GREEN}[2/2] Starting Frontend Server...${NC}"
echo -e "Directory: $FRONTEND_DIR"

# Check if node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}Warning: node_modules not found. Installing dependencies...${NC}"
    cd "$FRONTEND_DIR"
    yarn install
    cd "$SCRIPT_DIR"
fi

# Start frontend in background
(
    cd "$FRONTEND_DIR"
    echo -e "${GREEN}✓ Starting React development server on http://localhost:3000${NC}"
    yarn start
) &

FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}\n"

# Display status
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Both servers are running!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Backend:${NC}  http://localhost:8000"
echo -e "${GREEN}API Docs:${NC} http://localhost:8000/docs"
echo -e "${GREEN}Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}\n"

# Wait for both processes
wait
