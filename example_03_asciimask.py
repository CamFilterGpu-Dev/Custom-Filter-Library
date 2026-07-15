import sys
import cv2
import numpy as np
import customtkinter as ctk
import traceback

# ==============================================================================
# 🛠️ CAMFILTER CUSTOM SCRIPTING: ENVIRONMENT GUIDE 🛠️
# ==============================================================================
# This script runs directly inside the compiled CamfilterGPU environment. 
# You have native access to the bundled high-performance libraries.
#
# 📚 AVAILABLE LIBRARIES:
# - cv2 (OpenCV): Standard image manipulation and filtering.
# - numpy (np): Fast matrix operations.
# - customtkinter (ctk): For building embedded UIs.
# ==============================================================================

class CustomFilter:
    def __init__(self):
        self.name = "ASCII Mask Example"
        
        # New v96 GUI Variables
        self.char_sequence = " .,:-=+*c1s2z3u4o5v6e7a8x9n0w#m%W@"
        self.size = 6
        self.invert = False
        self.transparent = False
        self.bold = False  
        
        # State tracking for dynamic rebuilds
        self._last_chars = self.char_sequence
        self._last_size = self.size
        self._last_invert = self.invert
        self._last_bold = self.bold
        self.last_mirror_state = False  
        
        self.char_set = list(self.char_sequence)
        self.atlas = None
        self.atlas_mask = None
        self.tile_w = 10
        self.tile_h = 10
        
        self.build_atlas()

    def get_ui_schema(self):
        """v96 Architecture: Instructs the Node Editor to build the UI natively."""
        return [
            {"type": "text", "name": "char_sequence", "label": "Characters", "default": " .,:-=+*c1s2z3u4o5v6e7a8x9n0w#m%W@", "row": 0, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "size", "label": "Text Size (pt)", "min": 4, "max": 36, "default": 6, "row": 1, "column": 0, "columnspan": 4},
            {"type": "switch", "name": "invert", "label": "Reverse Colors", "default": False, "row": 2, "column": 0, "columnspan": 1},
            {"type": "switch", "name": "transparent", "label": "Transparent BG", "default": False, "row": 2, "column": 1, "columnspan": 1},
            {"type": "switch", "name": "bold", "label": "Bold Text", "default": False, "row": 2, "column": 2, "columnspan": 1}
        ]
        
    def check_state_and_rebuild(self):
        """Monitors v96 UI variables and rebuilds the atlas if they change."""
        current_chars = getattr(self, 'char_sequence', ".")
        if not current_chars: current_chars = "."
        
        changed = False
        if self._last_chars != current_chars:
            self._last_chars = current_chars
            self.char_set = list(current_chars)
            changed = True
            
        if self._last_size != getattr(self, 'size', 6):
            self._last_size = getattr(self, 'size', 6)
            self.size = int(self._last_size)
            changed = True
            
        if self._last_invert != getattr(self, 'invert', False):
            self._last_invert = getattr(self, 'invert', False)
            self.invert = self._last_invert
            changed = True
            
        if self._last_bold != getattr(self, 'bold', False):
            self._last_bold = getattr(self, 'bold', False)
            self.bold = self._last_bold
            changed = True
            
        if changed:
            self.build_atlas()
        
    def build_atlas(self):
        font = cv2.FONT_HERSHEY_PLAIN
        scale = self.size / 10.0
        
        if self.bold:
            thickness = max(2, int(self.size / 6))
        else:
            thickness = max(1, int(self.size / 12))
        
        max_w, max_h = 0, 0
        for c in self.char_set:
            (w, h), _ = cv2.getTextSize(c, font, scale, thickness)
            max_w, max_h = max(max_w, w), max(max_h, h)
            
        tile_w = max(2, max_w)
        tile_h = max(2, max_h + 2)
        
        bg_color = (0, 0, 0) if self.invert else (255, 255, 255)
        fg_color = (255, 255, 255) if self.invert else (0, 0, 0)
        
        num_chars = len(self.char_set)
        new_atlas = np.full((num_chars, tile_h, tile_w, 3), bg_color, dtype=np.uint8)
        new_mask = np.zeros((num_chars, tile_h, tile_w), dtype=np.uint8)
        
        is_mirrored = getattr(self, 'last_mirror_state', False)
        
        for i, c in enumerate(self.char_set):
            (w, h), _ = cv2.getTextSize(c, font, scale, thickness)
            x = (tile_w - w) // 2
            y = tile_h - 2
            
            temp_c = np.full((tile_h, tile_w, 3), bg_color, dtype=np.uint8)
            temp_m = np.zeros((tile_h, tile_w), dtype=np.uint8)
            
            cv2.putText(temp_c, c, (x, y), font, scale, fg_color, thickness, cv2.LINE_AA)
            cv2.putText(temp_m, c, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
            
            if is_mirrored:
                temp_c = cv2.flip(temp_c, 1)
                temp_m = cv2.flip(temp_m, 1)
                
            new_atlas[i] = temp_c
            new_mask[i] = temp_m
        
        self.atlas, self.atlas_mask, self.tile_w, self.tile_h = new_atlas, new_mask, tile_w, tile_h

    def apply(self, frame, mask, hands, eyes):
        # 1. Extreme Fallback safety to guarantee we never pass a dead frame
        if frame is None or frame.size == 0:
            return frame
            
        try:
            # v96 State Check: Rebuild atlas if the user adjusted the UI
            self.check_state_and_rebuild()
            
            # --- THREAD SAFETY: Grab frozen local copies of the atlas variables ---
            current_atlas = self.atlas
            current_atlas_mask = self.atlas_mask
            current_tw = self.tile_w
            current_th = self.tile_h

            current_mirror = False
            if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'app'):
                current_mirror = getattr(sys.modules['__main__'].app, 'is_mirrored', False)
                
            if current_mirror != getattr(self, 'last_mirror_state', False):
                self.last_mirror_state = current_mirror
                self.build_atlas()
                return frame
                
            if current_atlas is None or current_tw <= 0 or current_th <= 0:
                return frame
                
            cols = int(np.ceil(frame.shape[1] / current_tw))
            rows = int(np.ceil(frame.shape[0] / current_th))
            if cols <= 0 or rows <= 0: return frame
            
            # v96 Optimization: The mask is already uint8 (0-255). We can use blazing fast C++ bitwise operations
            # instead of slow 32-bit float math to isolate the foreground.
            isolated_foreground = cv2.bitwise_and(frame, frame, mask=mask)
            
            gray_fg = cv2.cvtColor(isolated_foreground, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray_fg, (cols, rows), interpolation=cv2.INTER_AREA)
            
            # --- THREAD SAFETY
            # Use the length of the frozen atlas, NEVER the live GUI variable
            num_chars = current_atlas.shape[0] 
            
            # v96 Optimization: Fast integer math for character mapping instead of floats
            indices = ((255 - small_gray).astype(np.uint16) * (num_chars - 1) // 255).astype(np.uint8)
            
            ascii_grid = current_atlas[indices]
            ascii_img = ascii_grid.transpose(0, 2, 1, 3, 4).reshape(rows * current_th, cols * current_tw, 3)
            ascii_full = ascii_img[0:frame.shape[0], 0:frame.shape[1]]
            
            # Convert 0-255 uint8 mask to binary hard mask for instant blitting
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            if getattr(self, 'transparent', False):
                # Map 0-255 mask to letter alpha indices using fast int math
                small_mask_resized = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)
                indices_mask = (small_mask_resized.astype(np.uint16) * (num_chars - 1) // 255).astype(np.uint8)
                
                mask_grid = current_atlas_mask[indices_mask]
                mask_img_raw = mask_grid.transpose(0, 2, 1, 3).reshape(rows * current_th, cols * current_tw)
                char_mask_full = mask_img_raw[0:frame.shape[0], 0:frame.shape[1]]
                
                # Combine the person mask and the character mask with C++ bitwise logic
                combined_mask = cv2.bitwise_and(binary_mask, char_mask_full)
                
                fg = cv2.bitwise_and(ascii_full, ascii_full, mask=combined_mask)
                bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(combined_mask))
                final_frame = cv2.add(fg, bg)
            else:
                fg = cv2.bitwise_and(ascii_full, ascii_full, mask=binary_mask)
                bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(binary_mask))
                final_frame = cv2.add(fg, bg)
            
            # 2. Strict dimension and formatting validation
            final_frame = final_frame.astype(np.uint8)
            
            if final_frame.size == 0 or len(final_frame.shape) != 3:
                return frame
            
            return final_frame
            
        except Exception as e:
            # If any numpy reshape math fails, it safely logs it to terminal and bypasses the effect
            print(f"ASCII Mask Math Error Bypassed: {e}")
            traceback.print_exc()
            return frame