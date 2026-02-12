#!/bin/bash

# Athena Clean Restart Script
# This script performs a clean restart of both frontend and backend

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                               ║${NC}"
echo -e "${BLUE}║        Athena Clean Restart Script            ║${NC}"
echo -e "${BLUE}║                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}⚠️  Killing processes on port $port...${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}✓ Port $port cleared${NC}"
    else
        echo -e "${GREEN}✓ Port $port is free${NC}"
    fi
}

# Function to clean directory
clean_directory() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo -e "${YELLOW}🧹 Cleaning $dir...${NC}"
        rm -rf "$dir"
        echo -e "${GREEN}✓ Cleaned $dir${NC}"
    fi
}

# Function to clean files by pattern
clean_files() {
    local pattern=$1
    local description=$2
    echo -e "${YELLOW}🧹 Cleaning $description...${NC}"
    find . -name "$pattern" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Cleaned $description${NC}"
}

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: Stopping Running Processes${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Kill processes on frontend port (3010)
kill_port 3010

# Kill processes on backend port (8010)
kill_port 8010

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Cleaning Backend${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$PROJECT_ROOT/backend"

# Clean Python cache
echo -e "${YELLOW}🧹 Cleaning Python cache files...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Python cache cleaned${NC}"

# Clean logs
if [ -d "logs" ]; then
    echo -e "${YELLOW}🧹 Cleaning backend logs...${NC}"
    rm -rf logs/*
    echo -e "${GREEN}✓ Backend logs cleaned${NC}"
fi

# Clean pytest cache
clean_directory ".pytest_cache"
clean_directory ".coverage"
clean_directory "htmlcov"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Cleaning Frontend${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$PROJECT_ROOT/frontend"

# Clean build artifacts
clean_directory "build"
clean_directory "dist"
clean_directory ".next"

# Clean cache directories
clean_directory ".cache"
clean_directory ".parcel-cache"
clean_directory ".eslintcache"

# Clean coverage reports
clean_directory "coverage"

# Clean temp files
echo -e "${YELLOW}🧹 Cleaning temp files...${NC}"
rm -f .DS_Store 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "*.log" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Temp files cleaned${NC}"

# Clean Yarn cache (optional - uncomment if needed)
# echo -e "${YELLOW}🧹 Cleaning Yarn cache...${NC}"
# yarn cache clean 2>/dev/null || true
# echo -e "${GREEN}✓ Yarn cache cleaned${NC}"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: Checking Dependencies${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    yarn install
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies present${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5: Creating .env files${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Create frontend .env if it doesn't exist
if [ ! -f ".env" ] && [ -f "env.txt" ]; then
    echo -e "${YELLOW}📝 Creating frontend .env from env.txt...${NC}"
    cp env.txt .env
    echo -e "${GREEN}✓ Frontend .env created${NC}"
else
    echo -e "${GREEN}✓ Frontend .env exists${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 6: Starting Services${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo -e "${GREEN}🚀 Starting Backend Server (Port 8010)...${NC}"
echo -e "${YELLOW}   Logs will be saved to: backend/logs/backend.log${NC}"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/backend/logs"

# Start backend in background
cd "$PROJECT_ROOT/backend"
nohup python3 run_server.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Wait a moment for backend to start
sleep 3

echo ""
echo -e "${GREEN}🚀 Starting Frontend Server (Port 3010)...${NC}"
echo -e "${YELLOW}   Logs will be saved to: frontend/logs/frontend.log${NC}"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/frontend/logs"

# Start frontend in background
cd "$PROJECT_ROOT/frontend"
nohup yarn start > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ Startup Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo -e "${GREEN}📍 Services Running:${NC}"
echo -e "   Backend:  http://localhost:8010 (PID: $BACKEND_PID)"
echo -e "   Frontend: http://localhost:3010 (PID: $FRONTEND_PID)"
echo -e "   API Docs: http://localhost:8010/docs"
echo ""
echo -e "${YELLOW}📋 Useful Commands:${NC}"
echo -e "   View backend logs:  ${BLUE}tail -f backend/logs/backend.log${NC}"
echo -e "   View frontend logs: ${BLUE}tail -f frontend/logs/frontend.log${NC}"
echo -e "   Stop backend:       ${BLUE}kill $BACKEND_PID${NC}"
echo -e "   Stop frontend:      ${BLUE}kill $FRONTEND_PID${NC}"
echo -e "   Stop all services:  ${BLUE}./stop.sh${NC}"
echo ""
echo -e "${GREEN}⏳ Waiting for services to be fully ready...${NC}"
sleep 5

# Health check
echo -e "${YELLOW}🔍 Performing health check...${NC}"
if curl -s http://localhost:8010/api/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend is responding${NC}"
else
    echo -e "${RED}⚠️  Backend may still be starting up. Check logs if needed.${NC}"
fi

echo ""
echo -e "${GREEN}🎉 All systems ready! Open http://localhost:3010 in your browser.${NC}"
echo ""
