#!/usr/bin/env python3
"""
Run the Streamlit version of the Data Analysis Toolkit

Usage:
    python run_streamlit.py [--port PORT] [--no-browser]
    
Or directly:
    streamlit run src/data_toolkit/streamlit_app.py

Arguments:
    --port PORT     Port to run on (default: 8501)
    --no-browser    Don't auto-open browser
    --debug         Enable debug mode
"""

import subprocess
import sys
import os
import argparse


def check_dependencies():
    """Check and install required dependencies"""
    required = ['streamlit', 'plotly', 'pandas', 'numpy', 'scikit-learn']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)


def main():
    parser = argparse.ArgumentParser(description='Launch Advanced Data Analysis Toolkit')
    parser.add_argument('--port', type=int, default=8501, help='Port to run on (default: 8501)')
    parser.add_argument('--no-browser', action='store_true', help="Don't auto-open browser")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "src", "data_toolkit", "streamlit_app.py")
    src_path = os.path.join(script_dir, "src")
    
    # Check if app exists
    if not os.path.exists(app_path):
        print(f"❌ Error: Could not find {app_path}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{src_path}:{pythonpath}" if pythonpath else src_path
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(args.port),
        "--browser.gatherUsageStats", "false"
    ]
    
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    # Display startup info
    print()
    print("=" * 60)
    print("🚀 Advanced Data Analysis Toolkit v9.1")
    print("=" * 60)
    print(f"📍 App path: {app_path}")
    print(f"🌐 URL: http://localhost:{args.port}")
    print(f"🛑 Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    # Run Streamlit
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    main()
