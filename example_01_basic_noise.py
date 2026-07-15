import cv2
import numpy as np

class CustomFilter:
    def __init__(self):
        self.name = "Basic: Noise & Invert"
        
        # UI Variables
        self.noise_amount = 0.2   # Default slider value
        self.invert_image = False # Default switch value
        self.color_noise = False  # Default to high-speed Black & White noise

    def get_ui_schema(self):
        """v96 Architecture: Defines the custom mini-GUI in the Node Editor dock."""
        return [
            {"type": "label", "name": "lbl_1", "label": "A basic tutorial filter template.", "font_size": 12, "row": 0, "column": 0, "columnspan": 4, "pady": 10},
            
            {"type": "slider", "name": "noise_amount", "label": "Noise Amount", "min": 0.0, "max": 1.0, "default": 0.2, "row": 1, "column": 0, "columnspan": 4},
            
            {"type": "gap", "size": 10, "row": 2, "column": 0, "columnspan": 4},
            
            {"type": "switch", "name": "invert_image", "label": "Invert Colors", "default": False, "row": 3, "column": 0, "columnspan": 2},
            {"type": "switch", "name": "color_noise", "label": "Color", "default": False, "row": 3, "column": 2, "columnspan": 2}
        ]

    def apply(self, frame, mask, hands, eyes):
        """Executes on every single video frame (e.g., 30 times a second)."""
        if frame is None or frame.size == 0:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # --- 1. INVERT EFFECT ---
        do_invert = getattr(self, 'invert_image', False)
        if do_invert:
            # Fast numpy slice [:, :, :3] only inverts Colors (BGR), protecting the Alpha channel
            result[:, :, :3] = cv2.bitwise_not(result[:, :, :3])
            
        # --- 2. NOISE EFFECT ---
        noise_val = float(getattr(self, 'noise_amount', 0.0))
        if noise_val > 0.0:
            std_dev = int(noise_val * 100)
            
            # Optimization 1: Calculate at exactly 50% resolution (75% fewer pixels, for speed!)
            half_w, half_h = w // 2, h // 2
            
            is_color = getattr(self, 'color_noise', False)
            
            if is_color:
                # 3-Channel Generation
                noise_half = np.empty((half_h, half_w, 3), dtype=np.uint8)
                cv2.randn(noise_half, (0, 0, 0), (std_dev, std_dev, std_dev))
            else:
                # Optimization 2: 1-Channel Generation (3x fewer random numbers generated)
                noise_1c = np.empty((half_h, half_w, 1), dtype=np.uint8)
                cv2.randn(noise_1c, 0, std_dev)
                # Instantly duplicate the single channel across B, G, and R
                noise_half = cv2.cvtColor(noise_1c, cv2.COLOR_GRAY2BGR)
                
            # Optimization 3: Nearest Neighbor is the absolute fastest way to upscale
            noise_full = cv2.resize(noise_half, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Add the fast-generated static directly to the BGR channels
            result[:, :, :3] = cv2.add(result[:, :, :3], noise_full)

        return result