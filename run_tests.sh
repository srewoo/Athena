#!/bin/bash
# Simple script to run tests from project root

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running Athena Test Suite${NC}\n"

# Check if we're in the right directory
if [ ! -d "tests" ]; then
    echo -e "${RED}Error: tests/ directory not found${NC}"
    echo -e "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at backend/venv${NC}"
    echo -e "Please run: cd backend && python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source backend/venv/bin/activate

# Install/update dependencies if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install -q -r backend/requirements.txt

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}\n"

if [ "$1" == "no-coverage" ]; then
    # Run without coverage (faster)
    pytest tests/ -v
elif [ "$1" == "debug" ]; then
    # Run with full debug output and coverage
    pytest tests/ --cov=project_api --cov=project_storage --cov=llm_client --cov=smart_test_generator --cov-report=html --cov-report=term -v -s --tb=long
    echo -e "\n${GREEN}Coverage report generated at htmlcov/index.html${NC}"
else
    # Default: run with coverage
    pytest tests/ --cov=project_api --cov=project_storage --cov=llm_client --cov=smart_test_generator --cov-report=html --cov-report=term -v
    echo -e "\n${GREEN}Coverage report generated at htmlcov/index.html${NC}"
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
else
    echo -e "\n${RED}Some tests failed. See output above.${NC}"
    exit 1
fi
