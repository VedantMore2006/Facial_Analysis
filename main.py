# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# Purpose: Application entry point that delegates to pipeline orchestrator
# Design: Keeps main.py minimal - all logic lives in modular components
# Usage: Run this file to start the facial analysis system
#        python main.py
# ============================================================================

"""  
Main entry point.
No logic should live here.
"""

from src.pipeline import run_pipeline

# Only run when executed directly (not when imported as module)
if __name__ == "__main__":
    # Delegate entire execution flow to pipeline orchestrator
    run_pipeline()