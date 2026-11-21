import cv2
import numpy as np
import time
import traceback


class CustomFilter:
    def __init__(self):
        self.name = "Iridescent Sparkle Aura"
        self._start_time = time.monotonic()

        # Cached resolution-dependent grids & sparkle field
        self._cached_shape = None
        self._coord_wave = None
        self._sparkle_x = None
        self._sparkle_y = None
        self._sparkle_phase = None
        self._sparkle_speed = None

    def release(self):
        pass

    def get_ui_schema(self):
        return [
            {"type": "label", "label": "--- Iridescent Aura Settings ---", "row": 0, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "intensity", "label": "Intensity", "min": 0, "max": 100, "default": 75, "row": 1, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "thickness", "label": "Thickness (px)", "min": 10, "max": 200, "default": 80, "row": 2, "column": 0, "columnspan": 4},
            {"type": "gap", "row": 3, "column": 0, "columnspan": 4}
        ]

    def _init_caches(self, h, w):
        """Precompute low-res wave coordinates and a pool of twinkling sparkle points."""
        sh, sw = max(1, h // 4), max(1, w // 4)
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)

        # Smooth diagonal + radial wave for organic iridescent color shifting
        self._coord_wave = (xx * 0.9 + yy * 0.6) + 18.0 * np.sin(yy * 0.08) + 14.0 * np.cos(xx * 0.07)

        # Pool of 500 potential sparkle coordinates across the frame
        num_sparkles = 500
        self._sparkle_x = np.random.randint(4, max(5, w - 4), size=num_sparkles)
        self._sparkle_y = np.random.randint(4, max(5, h - 4), size=num_sparkles)
        self._sparkle_phase = np.random.uniform(0.0, 2.0 * np.pi, size=num_sparkles).astype(np.float32)
        self._sparkle_speed = np.random.uniform(2.5, 6.5, size=num_sparkles).astype(np.float32)
        self._cached_shape = (h, w)

    def apply(self, frame, mask, hands_pointers, eyes):
        if frame is None or frame.size == 0 or mask is None:
            return frame

        try:
            h, w = frame.shape[:2]

            # 1. Thread-safe UI variable capture
            intensity = float(getattr(self, 'intensity', 75)) / 100.0
            thickness = float(getattr(self, 'thickness', 80))

            if intensity <= 0.01 or thickness < 2.0:
                return frame

            if self._cached_shape != (h, w):
                self._init_caches(h, w)

            t = time.monotonic() - self._start_time
            sh, sw = max(1, h // 4), max(1, w // 4)

            # 2. Fast 1/4-res Euclidean distance transform outside the subject profile
            mask_small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
            _, bin_small = cv2.threshold(mask_small, 127, 255, cv2.THRESH_BINARY)
            inv_small = cv2.bitwise_not(bin_small)

            dist_small = cv2.distanceTransform(inv_small, cv2.DIST_L2, 3)

            # Convert thickness to 1/4-res scale and build a smooth fade-out curve
            scaled_thick = max(1.0, thickness / 4.0)
            falloff_small = np.clip(1.0 - (dist_small / scaled_thick), 0.0, 1.0)
            falloff_small = np.power(falloff_small, 1.7)

            # 3. Animated iridescent rainbow field (computed at 1/4 res, then upscaled)
            hue_small = (self._coord_wave + dist_small * 2.2 - t * 38.0) % 180.0
            hsv_small = np.empty((sh, sw, 3), dtype=np.uint8)
            hsv_small[:, :, 0] = hue_small.astype(np.uint8)
            # Subtle iridescence: slightly softer saturation near the outer fade
            hsv_small[:, :, 1] = np.clip(165.0 + 90.0 * falloff_small, 0, 255).astype(np.uint8)
            hsv_small[:, :, 2] = 255

            rainbow_small = cv2.cvtColor(hsv_small, cv2.COLOR_HSV2BGR)
            rainbow_full = cv2.resize(rainbow_small, (w, h), interpolation=cv2.INTER_LINEAR)
            falloff_full = cv2.resize(falloff_small, (w, h), interpolation=cv2.INTER_LINEAR)

            # 4. Strictly mask out the interior of the subject profile (keeps soft RVM edges clean)
            outside_mask = (255.0 - mask.astype(np.float32)) / 255.0
            aura_alpha = np.clip(falloff_full * outside_mask * intensity, 0.0, 1.0)

            # 5. Blend the iridescent surround into the background
            alpha_3c = aura_alpha[:, :, np.newaxis]
            out = frame.astype(np.float32) * (1.0 - alpha_3c * 0.85) + rainbow_full.astype(np.float32) * (alpha_3c * 0.85)

            # 6. Time-dependent twinkling sparkles strictly inside the aura
            twinkle = np.sin(t * self._sparkle_speed + self._sparkle_phase)
            # Sharp sparkle peaks so only a fraction twinkle brightly at any moment
            bright = np.power(np.maximum(0.0, twinkle), 6)

            local_aura = aura_alpha[self._sparkle_y, self._sparkle_x]
            active = (local_aura > 0.08) & (bright > 0.25)

            if np.any(active):
                sparkle_layer = np.zeros((h, w, 3), dtype=np.uint8)
                xs = self._sparkle_x[active]
                ys = self._sparkle_y[active]
                strengths = bright[active] * local_aura[active]

                for x, y, s in zip(xs, ys, strengths):
                    val = int(min(255, 255 * s * 1.4))
                    col = (val, val, val)
                    arm = 3 if s > 0.55 else 2
                    cv2.line(sparkle_layer, (x - arm, y), (x + arm, y), col, 1, cv2.LINE_AA)
                    cv2.line(sparkle_layer, (x, y - arm), (x, y + arm), col, 1, cv2.LINE_AA)

                # Additively glow the sparkles over the aura
                out += sparkle_layer.astype(np.float32) * min(1.2, intensity * 1.2)

            return np.clip(out, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"Iridescent Aura Error Bypassed: {e}")
            traceback.print_exc()
            return frame