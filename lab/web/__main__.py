"""python -m lab.web [--port 8100]"""

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="pigeon lab website")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    print(f"pigeon lab on http://127.0.0.1:{args.port}")
    uvicorn.run("lab.web.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
