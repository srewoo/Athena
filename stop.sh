#!/bin/bash

# Athena Stop Script
# Gracefully stops both frontend and backend services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                               ║${NC}"
echo -e "${BLUE}║          Athena Stop Script                   ║${NC}"
echo -e "${BLUE}║                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    local service=$2
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}⚠️  Stopping $service (port $port)...${NC}"
        echo "$pids" | xargs kill -15 2>/dev/null || true
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if still running
        pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ ! -z "$pids" ]; then
            echo -e "${YELLOW}⚠️  Force stopping $service...${NC}"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
        
        echo -e "${GREEN}✓ $service stopped${NC}"
    else
        echo -e "${GREEN}✓ $service not running${NC}"
    fi
}

echo -e "${BLUE}Stopping services...${NC}"
echo ""

# Stop backend
kill_port 8010 "Backend"

# Stop frontend
kill_port 3010 "Frontend"

echo ""
echo -e "${GREEN}✨ All services stopped successfully!${NC}"
echo ""
