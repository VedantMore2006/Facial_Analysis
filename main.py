# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# Purpose: Application entry point that delegates to pipeline orchestrator
# Design: Keeps main.py minimal - all logic lives in modular components
# 
# Usage:
#   Live webcam:
#     python main.py
#   
#   Offline video file:
#     python main.py --video path/to/video.mp4
# ============================================================================

"""  
Main entry point.
Handles command-line argument parsing and frame source selection.
"""

import argparse
from src.pipeline import run_pipeline
from src.frame_source import CameraSource, VideoFileSource


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Facial Analysis Pipeline - Privacy-Safe Feature Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with live webcam (default)
  python main.py
  
  # Run with video file
  python main.py --video recordings/session1.mp4
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to video file for offline processing (if omitted, uses live webcam)'
    )
    
    return parser.parse_args()


def create_frame_source(args):
    """
    Create appropriate frame source based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        FrameSource instance (either CameraSource or VideoFileSource)
    """
    if args.video:
        # User specified video file - use offline processing
        print(f"🎥 Mode: Offline video processing")
        return VideoFileSource(args.video)
    else:
        # No video specified - use live webcam
        print(f"📹 Mode: Live webcam")
        return CameraSource()


# Only run when executed directly (not when imported as module)
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create appropriate frame source
    frame_source = create_frame_source(args)
    
    # Delegate execution flow to pipeline orchestrator
    run_pipeline(frame_source)