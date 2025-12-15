#!/bin/bash

# Test runner script with coverage reporting
# Usage: ./run_tests.sh [options]
#   -v, --verbose     : Run tests with verbose output
#   -c, --coverage    : Generate coverage report
#   -h, --html        : Generate HTML coverage report
#   -b, --backend-only: Run only backend tests (not repo-level tests)
#   --help            : Show this help message

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
VERBOSE=""
COVERAGE=""
HTML_COVERAGE=""
ALL_TESTS=true  # Changed to true - run all tests by default
TEST_PATH="test_endpoints.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=. --cov-report=term-missing"
            shift
            ;;
        -h|--html)
            HTML_COVERAGE="--cov=. --cov-report=html"
            COVERAGE="--cov=. --cov-report=term-missing"
            shift
            ;;
        -b|--backend-only)
            ALL_TESTS=false
            shift
            ;;
        --help)
            echo "Usage: ./run_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose        Run tests with verbose output"
            echo "  -c, --coverage       Generate coverage report"
            echo "  -h, --html           Generate HTML coverage report"
            echo "  -b, --backend-only   Run only backend tests (not repo-level tests)"
            echo "  --help               Show this help message"
            echo ""
            echo "Default: Runs ALL tests (backend + repo-level tests)"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests (default)"
            echo "  ./run_tests.sh -b                 # Run backend tests only"
            echo "  ./run_tests.sh -v                 # Run all tests with verbose output"
            echo "  ./run_tests.sh -c                 # Run all tests with coverage report"
            echo "  ./run_tests.sh -b -v              # Run backend tests with verbose"
            echo "  ./run_tests.sh -v -h              # Run all tests, verbose with HTML coverage"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}======================================${NC}"
if [ "$ALL_TESTS" = true ]; then
    echo -e "${YELLOW}  Running ALL Athena Tests${NC}"
else
    echo -e "${YELLOW}  Running Athena Backend Tests${NC}"
fi
echo -e "${YELLOW}======================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Please install test dependencies:"
    echo "  pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Determine test path
if [ "$ALL_TESTS" = true ]; then
    if [ -d "../tests" ]; then
        echo -e "${BLUE}Running backend tests + repo-level tests...${NC}"
        TEST_PATH="test_endpoints.py ../tests/"
    else
        echo -e "${YELLOW}Warning: ../tests/ directory not found, running backend tests only${NC}"
        TEST_PATH="test_endpoints.py"
    fi
else
    echo -e "${BLUE}Running backend tests only...${NC}"
    TEST_PATH="test_endpoints.py"
fi

echo ""

# Run tests
if [ -n "$HTML_COVERAGE" ]; then
    pytest $TEST_PATH $VERBOSE $HTML_COVERAGE

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}  Tests Passed! ✓${NC}"
        echo -e "${GREEN}======================================${NC}"
        echo ""
        echo -e "${YELLOW}HTML coverage report generated at: htmlcov/index.html${NC}"
        echo "Open it with: open htmlcov/index.html"
    else
        echo ""
        echo -e "${RED}======================================${NC}"
        echo -e "${RED}  Tests Failed! ✗${NC}"
        echo -e "${RED}======================================${NC}"
        exit 1
    fi
elif [ -n "$COVERAGE" ]; then
    pytest $TEST_PATH $VERBOSE $COVERAGE

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}  Tests Passed! ✓${NC}"
        echo -e "${GREEN}======================================${NC}"
    else
        echo ""
        echo -e "${RED}======================================${NC}"
        echo -e "${RED}  Some Tests Failed${NC}"
        echo -e "${RED}======================================${NC}"
        exit 1
    fi
else
    pytest $TEST_PATH $VERBOSE

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}  Tests Passed! ✓${NC}"
        echo -e "${GREEN}======================================${NC}"
    else
        echo ""
        echo -e "${RED}======================================${NC}"
        echo -e "${RED}  Some Tests Failed${NC}"
        echo -e "${RED}======================================${NC}"
        exit 1
    fi
fi
