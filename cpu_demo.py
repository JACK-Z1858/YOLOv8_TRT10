# SPDX-License-Identifier: AGPL-3.0-only

"""Minimal serial YOLO CPU baseline."""

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Ultralytics .pt model")
    parser.add_argument("source", help="Image, directory, video, camera, or stream")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    started = time.perf_counter()
    count = 0
    for _ in model.predict(source=args.source, device="cpu", batch=1, stream=True, verbose=False):
        count += 1
    elapsed = time.perf_counter() - started

    if count:
        print(f"frames: {count}")
        print(f"elapsed: {elapsed:.3f} s")
        print(f"average: {elapsed * 1000 / count:.3f} ms/frame")
        print(f"throughput: {count / elapsed:.3f} FPS")
    else:
        print("no frames processed")


if __name__ == "__main__":
    main()
