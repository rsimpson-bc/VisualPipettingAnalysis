"""
Optional live display window (OpenCV or Qt).
Stub — to be designed once the analysis pipeline is running.
"""


class LiveView:
    """Displays the current frame and annotated analysis results in real time."""

    def __init__(self):
        self._window_name = "PA Live View"

    def show(self, frame, annotations=None):
        """Update the display with a new frame and optional overlay annotations."""
        # TODO: implement with cv2.imshow or a Qt widget
        pass

    def close(self):
        pass
