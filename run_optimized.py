#!/usr/bin/env python3
"""
Optimized startup script for TheraVox AI
This script sets environment variables for better performance before starting the server
"""

import os
import sys
import logging

# Set environment variables for optimization
os.environ["OPENCV_THREADS"] = "2"  # Limit OpenCV threads
os.environ["AUDIO_TORCH_THREADS"] = "2"  # Limit PyTorch threads for audio
os.environ["TORCH_NUM_THREADS"] = "2"  # Limit overall PyTorch threads
os.environ["OMP_NUM_THREADS"] = "2"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "2"  # Limit MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # Limit NumExpr threads

# Disable some warnings for cleaner output
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Force CPU for audio processing if no CUDA (faster startup)
# os.environ["AUDIO_DEVICE"] = "cpu"  # Uncomment to force CPU

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Start the optimized TheraVox server"""
    print("🚀 Starting TheraVox AI with optimizations...")
    print("📊 Settings:")
    print(f"   - OpenCV threads: {os.environ.get('OPENCV_THREADS', 'default')}")
    print(f"   - PyTorch threads: {os.environ.get('TORCH_NUM_THREADS', 'default')}")
    print(f"   - Audio device: {os.environ.get('AUDIO_DEVICE', 'auto')}")
    print("   - Lazy loading enabled for all AI models")
    print()
    
    try:
        import uvicorn
        from main import app
        
        # Start the server with optimized settings
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            workers=1,  # Single worker for better resource management
            loop="asyncio",  # Use asyncio loop
            access_log=False,  # Disable access logging for better performance
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\\n👋 TheraVox server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())