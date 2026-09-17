import argparse
import numpy as np

from vision import Vision


def parse_args():
    p = argparse.ArgumentParser(description="Pen tracking with the D435i")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--record", metavar="FILE", help="record the stream to a .bag file")
    g.add_argument("--play", metavar="FILE", help="play back a .bag file instead of the camera")
    p.add_argument("--clip", type=float, default=1.0, help="clipping distance in meters")
    return p.parse_args()


def main():
    args = parse_args()

    with Vision(record_to=args.record, play_from=args.play,
                clipping_distance_m=args.clip) as vision:
        print(f"depth scale: {vision.depth_scale}")

        while True:
            if not vision.capture_frame():
                continue

            mask, res = vision.to_hsv()

            side_by_side = np.hstack((vision.background_removed(),
                                      vision.depth_colormap()))
            if vision.show(side_by_side):
                break

            if vision.show_hsv(mask, res):
                break


if __name__ == "__main__":
    main()