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
            