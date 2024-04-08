

import sys

def main():
    try:
        import small_fish.pipeline.main
    except ValueError as ve:
        raise ve

if __name__ == "__main__":
    sys.exit(main())