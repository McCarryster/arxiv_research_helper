import sys

def main():
    # run pytest programmatically. This requires pytest to be installed.
    # return code follows pytest convention (0 => success)
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Install it (e.g. pip install pytest) and re-run this script.")
        return 2

    # You can change args below to adjust which tests run; default is 'tests' directory.
    args = ["-q", "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/rag_and_graph/final/tests"]
    return_code = pytest.main(args)
    # propagate pytest exit code
    sys.exit(return_code)

if __name__ == "__main__":
    main()