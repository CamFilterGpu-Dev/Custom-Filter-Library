import cv2
import numpy as np
import os

ASSET_DIR = globals().get('ASSET_DIR', os.path.dirname(os.path.abspath(__file__)))

class CustomFilter:
    def __init__(self):
        self.name = "Sharpen All (Optimized)"
        # Standard variables
        self.amount = 0.5
        self.radius = 3.0
        self.threshold = 0.0
        
        # New Grid UI variables
        self.fast_mode = True
        self.invert = False
        self.mode = "Box"
        self.user_text = "Hello - the custom filter is running"
        self.text_color = "#D88E2E" # NEW: Default hex color for the text
        
        # Pre-allocate memory buffers
        self._blur_scratch = None  
        self._edge_scratch = None
        self._diff_scratch = None

    def get_ui_schema(self):
        return [
            # Sliders (Row 0, 1, 2)
            {"type": "slider", "name": "amount", "label": "Amount", "min": 0.0, "max": 4.0, "default": 0.5, "row": 0, "column": 0, "columnspan": 4, "height": 10, "pady": 5},
            {"type": "slider", "name": "radius", "label": "Radius", "min": 1.0, "max": 50.0, "default": 3.0, "row": 1, "column": 0, "columnspan": 4, "height": 10, "pady": 5},
            {"type": "slider", "name": "threshold", "label": "Threshold", "min": 0.0, "max": 50.0, "default": 0.0, "row": 2, "column": 0, "columnspan": 4, "height": 10, "pady": 5},
            
            # Two Side-by-Side Switches (Row 3)
            {"type": "switch", "name": "fast_mode", "label": "Force Fast Path", "default": True, "row": 3, "column": 0, "columnspan": 2},
            {"type": "switch", "name": "invert", "label": "Invert Output", "default": False, "row": 3, "column": 2, "columnspan": 2},
            
            # A Dropdown, Text Box, and Color Chooser (Row 4)
            {"type": "dropdown", "name": "mode", "label": "Blur Type", "options": ["Box", "Gaussian", "Median"], "default": "Box", "row": 4, "column": 0, "columnspan": 2},
            {"type": "text", "name": "user_text", "label": "Caption", "default": "Hello", "row": 4, "column": 2, "columnspan": 1},
            {"type": "color", "name": "text_color", "label": "Color", "default": "#FFB450", "row": 4, "column": 3, "columnspan": 1}
        ]

    def get_assets(self):
        return []

    def apply(self, frame, mask, hands, eyes):
        if self.amount <= 0.01:
            return frame

        # OPTIMIZATION 1: Ignore the Alpha Channel. 
        rgb = frame[:, :, :3]

        if self._blur_scratch is None or self._blur_scratch.shape != rgb.shape:
            self._blur_scratch = np.empty_like(rgb)
            self._edge_scratch = np.empty_like(rgb)
            self._diff_scratch = np.empty_like(rgb)
            
        k = max(3, int(self.radius) | 1) 
        
        # This is for a super-fast blurring step. OPTIMIZED for 1280x720.
        h, w = rgb.shape[:2]
        if self.mode in ["Gaussian", "Median"]:
            # Safe dynamic downscaling to prevent shape crash
            small_w, small_h = max(1, w // 4), max(1, h // 4) 
            bg_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            small_k = int(max(3, k // 4))
            blurred_small = cv2.blur(bg_small, (small_k, small_k))
            
            # Safely write the pixels into the pre-allocated scratch buffer
            self._blur_scratch[:] = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            cv2.boxFilter(rgb, -1, (k, k), dst=self._blur_scratch)
        
        A = self.amount * 2.0
        
        if self.threshold <= 0.1 or self.fast_mode:
            rgb_out = cv2.addWeighted(rgb, 1.0 + A, self._blur_scratch, -A, 0)
        else:
            cv2.subtract(rgb, self._blur_scratch, dst=self._edge_scratch)
            cv2.absdiff(rgb, self._blur_scratch, dst=self._diff_scratch)
            _, edge_mask = cv2.threshold(self._diff_scratch, self.threshold, 255, cv2.THRESH_BINARY)
            cv2.bitwise_and(self._edge_scratch, edge_mask, dst=self._edge_scratch)
            rgb_out = cv2.addWeighted(rgb, 1.0, self._edge_scratch, A, 0)

        if self.invert:
            cv2.bitwise_not(rgb_out, dst=rgb_out)
            
        # v96 Fix: Force the data back into the original 4-channel frame memory
        frame[:, :, :3] = rgb_out
            
        if self.user_text:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.0
            thickness = 1  # must be integer
            
            hex_str = getattr(self, "text_color", "#FFB450").lstrip('#')
            try:
                r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                bgr_color = (b, g, r)
            except Exception:
                bgr_color = (80, 180, 255) 
            
            h, w = frame.shape[:2]
            text_size, baseline = cv2.getTextSize(self.user_text, font, font_scale, thickness)
            tw, th = text_size
            text_x = (w - tw) // 2
            text_y = h - 30 
            
            if getattr(self, 'app_is_mirrored', True):
                # OPTIMIZATION 2: The "Tiny Box" Flip
                # We draw the text on a tiny blank array and flip that instead of the 720p frame!
                box_h, box_w = th + baseline + 10, tw + 10
                text_roi = np.zeros((box_h, box_w, 3), dtype=np.uint8)
                mask_roi = np.zeros((box_h, box_w), dtype=np.uint8)
                
                cv2.putText(text_roi, self.user_text, (5, th + 5), font, font_scale, bgr_color, thickness)
                cv2.putText(mask_roi, self.user_text, (5, th + 5), font, font_scale, 255, thickness)
                
                cv2.flip(text_roi, 1, text_roi)
                cv2.flip(mask_roi, 1, mask_roi)
                
                y1, x1 = text_y - th - 5, text_x - 5
                y2, x2 = y1 + box_h, x1 + box_w
                
                if y1 >= 0 and x1 >= 0 and y2 <= h and x2 <= w:
                    roi = frame[y1:y2, x1:x2]
                    # Paste instantly where the text mask exists
                    np.copyto(roi, text_roi, where=(mask_roi[:, :, None] > 0))
            else:
                cv2.putText(frame, self.user_text, (text_x, text_y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)

        return frame