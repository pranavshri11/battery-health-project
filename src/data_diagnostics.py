"""Compatibility entry point for the reusable dataset validator."""
try:
    from .data_preprocessing import main
except ImportError:
    from data_preprocessing import main

if __name__ == "__main__":
    main()
