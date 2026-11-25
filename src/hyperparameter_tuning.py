"""Run the shared battery-aware training/tuning entry point."""
try:
    from .model_training import main
except ImportError:
    from model_training import main

if __name__ == "__main__":
    main()
