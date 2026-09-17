import numpy as np
import cv2
import pyrealsense2 as rs


class Vision:
    def __init__(self, record_to=None, play_from=None, clipping_distance_m=2.0):
        """Configure the streams. Nothing starts until __enter__/start()."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.clipping_distance_in_meters = clipping_distance_m

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
        '''convert a frame of BGR to HSV frame'''
        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)

        lower_purple = np.array([125, 50, 50])
        upper_purple = ([165, 255, 255])

        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        res = cv2.bitwise_and(self.color_image, self.color_image, mask= mask)

        return mask, res
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

    def show_hsv(self, mask, res):
        '''show the hsv frame'''
        cv2.imshow('frame', self.color_image)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(1) & 0xFF
        return k == 27 or k == ord("q")
            