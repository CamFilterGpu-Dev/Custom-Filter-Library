import cv2
import numpy as np
import math

# ==============================================================================
# 🛠️ Perfunct CamFilterGPU: CUSTOM SCRIPTING GUIDE 🛠️
# ==============================================================================
# This script runs directly inside the compiled CamFilterGPU environment.
# You have native access to the bundled high-performance libraries.
# -NOTE, there is a lot more than just this. See the other examples for different controls
#  and tracking and image tools.
#
# 📚 AVAILABLE LIBRARIES:
# - cv2 (OpenCV): Standard image manipulation and filtering.
# - numpy (np): Fast matrix operations.
# - mediapipe: Custom pose, hand, or face-mesh tracking.
# - onnxruntime: For loading your own AI models (BYO-AI).
# - more coming as we add (and user's request)
#
# 🎛️ UI ARCHITECTURE (get_ui_schema):
# Make your own controls in the mini-gui...
# You can instantly generate sliders, switches, and text labels in the Node
# Editor by returning a list of dictionaries from `get_ui_schema()`. The app
# will automatically bind these UI elements to `self.variable_name`.
#
# 🚀 PERFORMANCE TIPS:
# - Pre-allocate or cache your numpy arrays in `__init__` or using `if not hasattr()`.
# - Use OpenCV (cv2) matrix functions instead of standard Python loops.
# - The `apply()` function runs ~30 times a second. Keep it lean!
# ==============================================================================

class CustomFilter:
    def __init__(self):
        # This name appears in the UI Node Editor
        self.name = "Example: Dithered Newsprint"
        
        # Default variables (will be overridden dynamically by UI sliders)
        self.dot_size = 4.0
        self.contrast = 1.0
        
        # Memory caching to avoid allocating heavy arrays every frame
        self._pattern = None
        self._last_dot_size = 0

    def get_ui_schema(self):
        """Defines the custom mini-GUI in the Node Editor dock."""
        return [
            {"type": "label", "name": "lbl_1", "label": "A fast Halftone/Newsprint effect.", "font_size": 12, "row": 0, "column": 0, "columnspan": 4, "pady": 10},
            {"type": "slider", "name": "dot_size", "label": "Dot Size", "min": 2.0, "max": 32.0, "default": 4.0, "row": 1, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "contrast", "label": "Contrast", "min": 0.5, "max": 2.0, "default": 1.0, "row": 2, "column": 0, "columnspan": 4}
        ]
        
    def apply(self, frame, mask, hands, eyes):
        """
        Executes ~30 times per second in the main render thread.
        
        PARAMETERS:
        - frame: Live BGR video image (numpy array)
        - mask: AI background segmentation mask (numpy array, 0.0 to 1.0)
        - hands: Tracked pointer fingers [(x, y, scale_distance), ...]
        - eyes: Tracked irises [(right_x, right_y), (left_x, left_y), glow_radius]
        """
        h, w = frame.shape[:2]

        # 1. Convert to grayscale (Fastest color space for threshold calculations)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Adjust contrast using fast linear scaling
        if self.contrast != 1.0:
            gray = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=128 * (1.0 - self.contrast))

        # 3. Create or fetch the halftone dot matrix pattern (Cached for extreme speed)
        current_dot = max(2, int(self.dot_size))
        if self._pattern is None or self._pattern.shape[:2] != (h, w) or self._last_dot_size != current_dot:
            # We use a 2D sine wave to create a perfect matrix of dots
            x = np.linspace(0, w, w, dtype=np.float32)
            y = np.linspace(0, h, h, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            
            # The math generates values from 0 to 255 based on the sine peaks
            pattern = (np.sin(X * (np.pi / current_dot)) * np.sin(Y * (np.pi / current_dot)) * 127.5 + 127.5)
            self._pattern = pattern.astype(np.uint8)
            self._last_dot_size = current_dot

        # 4. Apply the dither threshold
        # cv2.compare instantly checks if the camera pixel is brighter than the dot matrix pattern
        # It returns 255 (White) if True, and 0 (Black) if False. Zero Python loops required!
        newsprint = cv2.compare(gray, self._pattern, cv2.CMP_GT)

        # 5. Interactive Element: Draw an indicator on tracked fingers
        if hands:
            for hnd in hands:
                fx, fy, scale = hnd
                # Draw a gray circle over the finger
                cv2.circle(newsprint, (int(fx), int(fy)), int(max(10, scale)), 128, -1)

        # MUST return a valid 3-channel BGR numpy array
        return cv2.cvtColor(newsprint, cv2.COLOR_GRAY2BGR)
