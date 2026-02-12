#!/bin/bash

# Athena Restart Script
# Stops and restarts both services without cleaning

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                               ║${NC}"
echo -e "${BLUE}║         Athena Quick Restart                  ║${NC}"
echo -e "${BLUE}║                                               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}1. Stopping services...${NC}"
"$SCRIPT_DIR/stop.sh"

echo ""
echo -e "${GREEN}2. Starting services...${NC}"
sleep 2

# Run the start script without cleaning
"$SCRIPT_DIR/start_backend.sh" &
sleep 3
"$SCRIPT_DIR/start_frontend.sh" &

echo ""
echo -e "${GREEN}✨ Restart initiated!${NC}"
echo -e "   Backend:  http://localhost:8010"
echo -e "   Frontend: http://localhost:3010"
echo ""
