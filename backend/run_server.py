#!/usr/bin/env python3
"""
Athena Backend Server Runner
Starts the FastAPI application on configured port
"""
import uvicorn
import os
import multiprocessing
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8010))
    host = os.getenv('HOST', '0.0.0.0')

    # Worker configuration
    # Default: use number of CPU cores, max 4 for reasonable resource usage
    cpu_count = multiprocessing.cpu_count()
    default_workers = min(cpu_count, 4)
    workers = int(os.getenv('WORKERS', default_workers))

    # Development mode (reload) only works with 1 worker
    reload = os.getenv('RELOAD', 'false').lower() == 'true'
    if reload:
        workers = 1
        print(f"⚠️  Running in DEVELOPMENT mode (reload enabled, 1 worker)")
    else:
        print(f"🏭 Running in PRODUCTION mode ({workers} workers)")

    print(f"🚀 Starting Athena Backend Server")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"👷 Workers: {workers}")
    print(f"💻 CPU Cores: {cpu_count}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔄 ReDoc: http://localhost:{port}/redoc")
    print("")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=100,
        backlog=2048
    )
