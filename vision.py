import numpy as np
import cv2
import pyrealsense2 as rs


class Vision:
    def __init__(self, record_to=None, play_from=None, clipping_distance_m=2.0):
        """Configure the streams. Nothing starts until __enter__/start()."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.clipping_distance_in_meters = clipping_distance_m

        # HSV bounds for the pen. OpenCV hue runs 0-179, not 0-359, so these
        # are half what a color picker would report. Tune them per lighting.
        self.lower_purple = np.array([111, 136, 92])
        self.upper_purple = np.array([143, 255, 255])

        self.playback = play_from is not None
        if self.playback:
            # A .bag file already contains the stream settings, so do not
            # enable_stream here
            self.config.enable_device_from_file(play_from, repeat_playback=False)
        else:
            # resolve() peeks at the device without starting it
            wrapper = rs.pipeline_wrapper(self.pipeline)
            device = self.config.resolve(wrapper).get_device()
            self._check_rgb(device)
            if record_to is not None:
                self.config.enable_record_to_file(record_to)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Built once, reused for every frame
        self.align = rs.align(rs.stream.color)

        # Filled in by start()
        self.profile = None
        self.depth_scale = None
        self.clipping_distance = None
        self.intrinsics = None

        # Filled in by capture_frame()
        self.depth_frame = None
        self.color_frame = None
        self.depth_image = None   # uint16, (480, 640), raw depth units
        self.color_image = None   # uint8,  (480, 640, 3), BGR

    # ---------- in loop ----------

    @staticmethod
    def _check_rgb(device):
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                return
        raise RuntimeError("This requires a depth camera with a color sensor")

    def start(self):
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()   # meters per depth unit
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        return self

    def stop(self):
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        cv2.destroyAllWindows()
        return False

    # ---------- capture ----------

    def capture_frame(self, timeout_ms=5000):
        """Grab one aligned pair. Returns True on success, False if the frame was incomplete"""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return False

        # Keep the frame objects alive: the numpy arrays below are views into
        # librealsense's buffers, not copies.
        self.depth_frame = depth_frame
        self.color_frame = color_frame
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        return True

    def to_hsv(self):
        '''convert a frame of BGR to HSV, and threshold it for purple'''
        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_purple, self.upper_purple)

        res = cv2.bitwise_and(self.color_image, self.color_image, mask=mask)

        return hsv, mask, res

    # ---------- threshold tuning ----------

    TUNER = "tuner"

    # (slider label, which bound, which channel, slider max)
    _SLIDERS = (
        ("H min", "lower", 0, 179),   # OpenCV hue is 0-179
        ("H max", "upper", 0, 179),
        ("S min", "lower", 1, 255),
        ("S max", "upper", 1, 255),
        ("V min", "lower", 2, 255),
        ("V max", "upper", 2, 255),
    )

    def create_tuner(self):
        """Build the HSV slider window. Call once, before the loop."""
        cv2.namedWindow(self.TUNER, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TUNER, 460, 130)
        for label, bound, channel, maxval in self._SLIDERS:
            start = (self.lower_purple if bound == "lower" else self.upper_purple)[channel]
            # The callback is required but unused -- positions are polled in
            # read_tuner() instead, so the sliders stay in sync with the loop.
            cv2.createTrackbar(label, self.TUNER, int(start), maxval, lambda _v: None)

    def read_tuner(self):
        """Copy the slider positions into lower_purple / upper_purple."""
        lower, upper = np.zeros(3, np.int32), np.zeros(3, np.int32)
        for label, bound, channel, _maxval in self._SLIDERS:
            target = lower if bound == "lower" else upper
            target[channel] = cv2.getTrackbarPos(label, self.TUNER)
        self.lower_purple, self.upper_purple = lower, upper

    def show_tuner(self, mask):
        """Draw the current bounds and match count into the tuner window."""
        lo, hi = self.lower_purple, self.upper_purple
        matched = int(np.count_nonzero(mask))
        panel = np.zeros((130, 460, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"H {lo[0]:3d} - {hi[0]:3d}",
            f"S {lo[1]:3d} - {hi[1]:3d}",
            f"V {lo[2]:3d} - {hi[2]:3d}",
            f"matched {matched:6d} px  ({100.0 * matched / mask.size:.2f}%)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (12, 28 + i * 28), font, 0.6,
                        (180, 120, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.TUNER, panel)

    def tuned_bounds(self):
        """The current bounds as a paste-ready snippet for __init__."""
        lo, hi = self.lower_purple, self.upper_purple
        return (f"self.lower_purple = np.array([{lo[0]}, {lo[1]}, {lo[2]}])\n"
                f"self.upper_purple = np.array([{hi[0]}, {hi[1]}, {hi[2]}])")
    
    # ---------- derived images ----------

    def background_removed(self, grey_color=153):
        """Color image with everything past the clipping distance greyed out."""
        depth_3d = np.dstack((self.depth_image,) * 3)
        return np.where(
            (depth_3d > self.clipping_distance) | (depth_3d <= 0),
            grey_color,
            self.color_image,
        ).astype(np.uint8)

    def depth_colormap(self):
        scaled = cv2.convertScaleAbs(self.depth_image, alpha=0.03)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    def find_contours(self, mask=None, min_area=100):
        """Outer contours of the purple regions, largest first.

        Pass the mask from to_hsv() to reuse it; otherwise the HSV conversion
        and threshold get redone here, doubling the per-frame work.
        Contours smaller than min_area are dropped so sensor speckle does not
        cover the frame in tiny outlines. Set min_area=0 to keep everything.
        """
        if mask is None:
            _, mask, _ = self.to_hsv()

        # CHAIN_APPROX_NONE, not SIMPLE: SIMPLE collapses straight runs down to
        # their endpoints, which can leave a big blob with only 4 points --
        # below the 5 that fitEllipse requires.
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if min_area:
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        return sorted(contours, key=cv2.contourArea, reverse=True)

    # ---------- the pen ----------

    def find_pen(self, mask=None, min_area=50, close_px=9):
        """One ellipse around all the visible purple, plus its 3D center.

        Returns None if nothing qualifies, otherwise a dict with:
            ellipse   ((cx, cy), (minor, major), angle) -- pass to cv2.ellipse
            center    (cx, cy) in color-image pixels
            depth_m   median depth over the purple pixels, in meters
            point     (x, y, z) in meters in the camera frame, or None
            length_px, width_px, angle
        """
        if mask is None:
            _, mask, _ = self.to_hsv()

        # Keep the unbridged mask: the 3D work below must only sample pixels
        # that are really purple, not the background inside a bridged gap.
        purple_only = mask

        # Glare along the barrel and fingers gripping it split the pen into
        # several blobs. Closing bridges those gaps so they read as one object
        # instead of each getting its own ellipse.
        if close_px:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours = self.find_contours(mask, min_area)
        if not contours:
            return None

        # Pool every surviving contour's points and fit a single ellipse to the
        # lot, so the result spans the whole visible pen rather than one piece.
        points = np.vstack(contours)
        if len(points) < 5:
            return None
        (cx, cy), (minor, major), angle = cv2.fitEllipse(points)

        # Depth is sampled over the purple pixels only. A bounding box around a
        # thin diagonal pen is mostly background, so its median would report the
        # desk behind the pen instead of the pen.
        depth_m = self._median_depth(mask)

        point = None
        if depth_m > 0:
            # intrinsics are the COLOR stream's, and align.process() already put
            # depth in the color camera's frame, so pixel and depth agree here.
            point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], depth_m)

        # Orientation from the 3D shape of the barrel itself.
        centroid_3d, axis = self.pen_axis(purple_only)
        rpy = self.axis_to_rpy(axis) if axis is not None else None

        return {
            "ellipse": ((cx, cy), (minor, major), angle),
            "center": (cx, cy),
            "depth_m": depth_m,
            "point": point,             # deprojected ellipse center
            "centroid_3d": centroid_3d,  # mean of the pen's actual 3D points
            "axis": axis,               # unit vector along the barrel
            "rpy": rpy,                 # (roll, pitch, yaw) in degrees
            "width_px": minor,
            "length_px": major,
            "angle": angle,
        }

    def points_3d(self, region_mask):
        """Nx3 array of camera-frame points for the valid pixels under a mask."""
        ys, xs = np.nonzero((region_mask > 0) & (self.depth_image > 0))
        if xs.size == 0:
            return np.empty((0, 3))
        z = self.depth_image[ys, xs].astype(np.float64) * self.depth_scale
        i = self.intrinsics
        # This camera reports all-zero distortion coefficients for the color
        # stream, so the plain pinhole formula is exactly what
        # rs2_deproject_pixel_to_point would compute -- and doing it as array
        # math beats calling that per pixel a few thousand times a frame.
        x = (xs - i.ppx) / i.fx * z
        y = (ys - i.ppy) / i.fy * z
        return np.column_stack((x, y, z))

    def pen_axis(self, region_mask, min_points=30, min_elongation=3.0):
        """(centroid, unit axis) of the pen in 3D, or (None, None).

        The axis is the first principal component of the barrel's 3D points:
        the direction along which the pen is longest.

        Returns (None, None) when the point cloud is not clearly elongated.
        That happens when the pen points nearly straight at the camera -- it
        then projects to a blob with no visible length, and the largest
        variance direction is noise rather than the barrel.
        """
        pts = self.points_3d(region_mask)
        if len(pts) < min_points:
            return None, None

        centroid = pts.mean(axis=0)
        # Rows of vt are the principal directions, already unit length and
        # ordered by variance, so vt[0] is the barrel's direction. The singular
        # values say how elongated the cloud actually is along each.
        _, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        if s[1] <= 0 or s[0] / s[1] < min_elongation:
            return None, None
        axis = vt[0]

        # A line has two opposite directions and nothing here identifies the
        # tip, so canonicalise to the half pointing away from the camera. Doing
        # this on z (rather than x) keeps pitch and yaw continuous for a pen
        # anywhere in front of the camera; the seam sits at yaw = +/-90, i.e. a
        # pen pointing exactly sideways.
        if axis[2] < 0:
            axis = -axis
        return centroid, axis

    @staticmethod
    def axis_to_rpy(axis):
        """(roll, pitch, yaw) in degrees for a direction in the camera frame.

        Camera frame is +x right, +y down, +z forward.
        Roll is rotation about the pen's own long axis. A round barrel looks
        identical at every roll angle, so it is unobservable and returned as
        0.0 rather than a made-up number.
        """
        dx, dy, dz = axis
        yaw = float(np.degrees(np.arctan2(dx, dz)))
        pitch = float(np.degrees(-np.arcsin(np.clip(dy, -1.0, 1.0))))
        roll = 0.0
        return roll, pitch, yaw

    def _median_depth(self, region_mask):
        """Median depth in meters over a mask, ignoring invalid (zero) pixels."""
        valid = self.depth_image[(region_mask > 0) & (self.depth_image > 0)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid)) * self.depth_scale

    @staticmethod
    def print_center(pen, inline=True):
        """Print the pen's 3D center as (x, y, z) in meters, camera frame.

        The loop runs at 30 Hz, so by default this rewrites one line rather
        than scrolling. Pass inline=False for one line per frame, which is what
        you want when piping to a file.
        """
        if pen is None or pen["point"] is None:
            text = "pen center: --"
        else:
            x, y, z = pen["point"]
            text = f"pen center: ({x:+.4f}, {y:+.4f}, {z:+.4f}) m"

        # Pad to overwrite whatever the previous, possibly longer, line left.
        print(f"\r{text:<44}", end="" if inline else "\n", flush=True)

    def draw_pen(self, pen, color=(0, 220, 0), thickness=2):
        """Aligned color image with the pen ellipse, its center, and 3D position."""
        canvas = self.color_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        if pen is None:
            cv2.putText(canvas, "no pen", (10, 25), font, 0.6,
                        (180, 180, 180), 1, cv2.LINE_AA)
            return canvas

        cv2.ellipse(canvas, pen["ellipse"], color, thickness)
        cx, cy = int(round(pen["center"][0])), int(round(pen["center"][1]))
        cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

        if pen["point"] is None:
            label = "no depth at center"
        else:
            x, y, z = pen["point"]
            label = f"xyz {x:+.3f} {y:+.3f} {z:.3f} m"
        cv2.putText(canvas, label, (10, 25), font, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{pen['length_px']:.0f}x{pen['width_px']:.0f} px"
                            f"  {pen['angle']:.0f} deg",
                    (10, 50), font, 0.5, color, 1, cv2.LINE_AA)

        if pen["rpy"] is not None:
            roll, pitch, yaw = pen["rpy"]
            cv2.putText(canvas, f"pitch {pitch:+.0f}  yaw {yaw:+.0f}  (roll n/a)",
                        (10, 72), font, 0.5, color, 1, cv2.LINE_AA)

            # Project the 3D axis back to pixels as a sanity check: if the blue
            # line does not lie along the barrel, the depth on the pen is bad.
            centroid, axis = pen["centroid_3d"], pen["axis"]
            half = 0.5 * pen["length_px"] / self.intrinsics.fx * centroid[2]
            ends = [rs.rs2_project_point_to_pixel(self.intrinsics, list(centroid + s * half * axis))
                    for s in (-1.0, 1.0)]
            (ax, ay), (bx, by) = ends
            cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (255, 160, 0), 2)
        return canvas

    def draw_boundary(self, mask=None, min_area=100, color=(0, 220, 0),
                      thickness=2, shape="ellipse"):
        """The aligned color image with the purple regions outlined.

        shape="contour" traces the exact blob outline.
        shape="ellipse" fits an ellipse to each blob instead, which gives a
        clean shape plus an orientation -- useful for something pen-shaped.
        """
        contours = self.find_contours(mask, min_area)

        # copy() matters: color_image is a view into librealsense's buffer, so
        # drawing in place would corrupt the frame that background_removed()
        # and any active recording read from.
        canvas = self.color_image.copy()

        if shape == "ellipse":
            for c in contours:
                # fitEllipse needs 5+ points and throws below that.
                if len(c) >= 5:
                    cv2.ellipse(canvas, cv2.fitEllipse(c), color, thickness)
        else:
            cv2.drawContours(canvas, contours, -1, color, thickness)
        return canvas

    # ---------- geometry ----------

    def depth_at(self, px, py):
        """Depth in meters at a pixel of the color image."""
        return self.depth_frame.get_distance(int(px), int(py))

    def pixel_to_point(self, px, py, depth_m=None):
        """(x, y, z) in meters in the camera frame for a color-image pixel."""
        if depth_m is None:
            depth_m = self.depth_at(px, py)
        if depth_m == 0:
            return None   # no depth reading there
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [px, py], depth_m)

    # ---------- display ----------

    @staticmethod
    def show(image, window="vision"):
        """Draw one frame. Returns True if the user asked to quit."""
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, image)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q") or key == 27

    def show_hsv(self, hsv, mask, res):
        '''show the hsv frame'''
        for window, image in (("hsv", hsv), ("mask", mask), ("res", res)):
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.imshow(window, image)
        k = cv2.waitKey(1) & 0xFF
        return k == 27 or k == ord("q")
            