"""
Main entry point for the RAG Pipeline application.
Handles CLI arguments and starts the FastAPI server.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

import uvicorn

# Add the parent directory to sys.path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging(log_level: str = "info"):
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    
    # Application configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for data storage and caching"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    
    # Development options
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    
    # Initialization options
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip pipeline initialization on startup (for faster testing)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Creating data directory: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)
    
    # Check port range
    if not (1 <= args.port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got {args.port}")
    
    return True


def main():
    """Main entry point."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Starting RAG Pipeline API Server")
        logger.info(f"Configuration:")
        logger.info(f"  Host: {args.host}")
        logger.info(f"  Port: {args.port}")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Model: {args.model_name}")
        logger.info(f"  Reload: {args.reload}")
        logger.info(f"  Log level: {args.log_level}")
        
        # Set environment variables for the API
        os.environ["RAG_DATA_DIR"] = args.data_dir
        os.environ["RAG_MODEL_NAME"] = args.model_name
        
        if args.skip_init:
            os.environ["RAG_SKIP_INIT"] = "true"
            logger.info("⚠️  Pipeline initialization will be skipped")
        
        # Start the server
        logger.info(f"🌐 Starting server on http://{args.host}:{args.port}")
        logger.info(f"📚 API documentation available at http://{args.host}:{args.port}/docs")
        
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)


def print_usage_examples():
    """Print usage examples."""
    examples = """
Usage Examples:
    
    # Basic usage (default settings)
    python main.py
    
    # Custom host and port
    python main.py --host 0.0.0.0 --port 8080
    
    # Development mode with auto-reload
    python main.py --reload --log-level debug
    
    # Custom data directory and model
    python main.py --data-dir /path/to/data --model-name sentence-transformers/all-mpnet-base-v2
    
    # Skip initialization for faster testing
    python main.py --skip-init
    
After starting, visit:
    - http://localhost:8000 - API root
    - http://localhost:8000/docs - Interactive API documentation
    - http://localhost:8000/health - Health check
    """
    print(examples)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        print_usage_examples()
    else:
        main()
