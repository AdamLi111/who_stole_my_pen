import argparse
import numpy as np

from vision import Vision


def parse_args():
    p = argparse.ArgumentParser(description="Pen tracking with the D435i")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--record", metavar="FILE", help="record the stream to a .bag file")
    g.add_argument("--play", metavar="FILE", help="play back a .bag file instead of the camera")
    p.add_argument("--clip", type=float, default=1.0, help="clipping distance in meters")
    p.add_argument("--tune", action="store_true",
                   help="show HSV sliders for tuning the purple range")
    return p.parse_args()


def main():
    args = parse_args()

    with Vision(record_to=args.record, play_from=args.play,
                clipping_distance_m=args.clip) as vision:
        print(f"depth scale: {vision.depth_scale}")

        if args.tune:
            vision.create_tuner()

        while True:
            if not vision.capture_frame():
                continue

            if args.tune:
                # Pick up slider movement before thresholding this frame.
                vision.read_tuner()

            hsv, mask, res = vision.to_hsv()

            if args.tune:
                vision.show_tuner(mask)

            # One ellipse around the whole visible pen, with its 3D center.
            # Reuse the mask we already computed instead of redoing to_hsv().
            pen = vision.find_pen(mask)
            vision.print_center(pen)
            if vision.show(vision.draw_pen(pen), window="pen"):
                break

            side_by_side = np.hstack((vision.background_removed(),
                                      vision.depth_colormap()))
            if vision.show(side_by_side):
                break

            if vision.show_hsv(hsv, mask, res):
                break
            

        print()   # leave the rewriting readout line behind

        if args.tune:
            print("Tuned bounds -- paste into Vision.__init__:\n")
            print(vision.tuned_bounds())


if __name__ == "__main__":
    main()