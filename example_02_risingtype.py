import cv2
import numpy as np
import keyboard
import random

class CustomFilter:
    def __init__(self):
        self.name = "Rising Type: Floating letters"
        
        self.current_text = ""
        self.floating_chars = []
        
        # v96 GUI Variables
        self.font_face = "Simplex"
        self.text_color = "#7A9DB5"  # Default Gray-Blue
        self.is_italic = False
        self.font_scale = 1.0
        self.font_thick = 2
        self.fade_altitude = 290  # Default: 290px of travel (fades at halfway point Y=360)
        self.float_speed = 1.0    # Default: 1.0x multiplier
        
        # Safely track just THIS script's hook so we don't break the main app
        self.hook = keyboard.on_press(self.on_key)
        
    def release(self):
        """v96 Node Editor cleanup: Crucial to prevent zombie keystrokes on hot-reload."""
        try:
            keyboard.unhook(self.hook)
        except:
            pass

    def get_ui_schema(self):
        """v96 Architecture: Native GUI for font selection."""
        return [
            {"type": "label", "name": "instruction_lbl", "label": "Type anything and hit <Enter> ...", "text_color": "#FFFFFF", "font_size": 14, "row": 0, "column": 0, "columnspan": 4, "pady": 10},
            {"type": "dropdown", "name": "font_face", "label": "Font Face", "options": ["Simplex", "Plain", "Duplex", "Complex", "Triplex", "Script Simplex", "Script Complex"], "default": "Simplex", "row": 1, "column": 0, "columnspan": 2},
            {"type": "color", "name": "text_color", "label": "Color", "default": "#7A9DB5", "row": 1, "column": 2, "columnspan": 1},
            {"type": "switch", "name": "is_italic", "label": "Italic", "default": False, "row": 1, "column": 3, "columnspan": 1},
            {"type": "gap", "size": 20, "row": 2, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "font_scale", "label": "Font Scale", "min": 0.5, "max": 3.0, "default": 1.0, "row": 3, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "font_thick", "label": "Thickness", "min": 1, "max": 5, "default": 2, "row": 4, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "fade_altitude", "label": "Float Height", "min": 50, "max": 700, "default": 290, "row": 5, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "float_speed", "label": "Float Speed", "min": 0.1, "max": 8.0, "default": 1.0, "row": 6, "column": 0, "columnspan": 4}
        ]
        
    def get_current_font(self):
        """Maps the UI string back to OpenCV's internal C++ font constants."""
        fonts = {
            "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
            "Plain": cv2.FONT_HERSHEY_PLAIN,
            "Duplex": cv2.FONT_HERSHEY_DUPLEX,
            "Complex": cv2.FONT_HERSHEY_COMPLEX,
            "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
            "Script Simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            "Script Complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        }
        
        face = getattr(self, 'font_face', "Simplex")
        base_font = fonts.get(face, cv2.FONT_HERSHEY_SIMPLEX)
        
        if getattr(self, 'is_italic', False):
            base_font |= cv2.FONT_ITALIC
            
        return base_font
        
    def on_key(self, event):
        """Processes keystrokes in the background."""
        if event.name == 'space':
            self.current_text += ' '
        elif event.name == 'backspace':
            self.current_text = self.current_text[:-1]
        elif event.name == 'enter':
            self.launch_text()
        elif len(event.name) == 1:
            # Ignore modifiers like 'shift' or 'ctrl', just grab the letter
            self.current_text += event.name
            
    def launch_text(self):
        """Calculates velocities and starting coordinates when Enter is pressed."""
        if not self.current_text: return
        
        # Pull dynamic font settings from the live GUI variables
        font = self.get_current_font()
        
        # v96 Fix: Force strict casting to prevent OpenCV from crashing on string/float UI values
        scale = float(getattr(self, 'font_scale', 1.0))
        # Ensure thickness never drops to 0
        thickness = max(1, int(float(getattr(self, 'font_thick', 2.1))))
        
        measure_text = self.current_text.replace(' ', 'a')
        (tw, th), _ = cv2.getTextSize(measure_text, font, scale, thickness)
        # Force strict python int casting (Numpy types can silently crash cv2.putText)
        cx = int((1280 - tw) // 2)
        
        for i, char in enumerate(self.current_text):
            prefix = self.current_text[:i].replace(' ', 'a')
            (pw, ph), _ = cv2.getTextSize(prefix, font, scale, thickness)
            start_x = cx + pw
            
            # 2. ALPHABETICAL TWEAK: Calculate the base factor, but we will use it less
            val = ord(char.lower()) if char.isalpha() else 100
            speed_factor = ((val - 97) / 25.0) if 97 <= val <= 122 else 0.5
            
            # 3. RANDOM JITTER: Reduced alphabetical impact to 8.0, added random float
            # Now, even if you type '========', they will drift at slightly different speeds
            random_variance = random.uniform(-2.5, 3.5) * 0.2
            velocity = 2.0 + (speed_factor * 2.0) + random_variance
            
            self.floating_chars.append({
                'char': char,
                'x': start_x,
                'y': 650,
                'vel': velocity
            })
            
        self.current_text = ""
        
    def apply(self, frame, mask, hands, eyes):
        """Executes 30 times a second. Renders the ghostly typing effect."""
        if frame is None or frame.size == 0:
            return frame
            
        final_frame = frame.copy()
        text_layer = np.zeros_like(final_frame)
        
        # Pull font settings from GUI (MUST match what launch_text used)
        font = self.get_current_font()
        
        # v96 Fix: Force strict casting to prevent OpenCV from crashing
        scale = float(getattr(self, 'font_scale', 1.0))
        thickness = max(1, int(float(getattr(self, 'font_thick', 2))))
        
        # Convert hex color to BGR tuple safely
        hex_color = getattr(self, 'text_color', '#7A9DB5').lstrip('#')
        try:
            base_r, base_g, base_b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            base_r, base_g, base_b = (180, 180, 180) # Fallback gray
            
        static_color = (base_b, base_g, base_r, 255)
        
        # Draw the active typing text onto the blank layer
        if self.current_text:
            measure_text = self.current_text.replace(' ', 'a')
            (tw, th), _ = cv2.getTextSize(measure_text, font, scale, thickness)
            cx = int((1280 - tw) // 2)
            cv2.putText(text_layer, self.current_text, (cx, 650), font, scale, static_color, thickness, cv2.LINE_AA)
            
        # Update and draw the floating characters onto the blank layer
        active_chars = []
        
        # v96 Fix: Live speed multiplier safely cast from UI
        speed_mult = float(getattr(self, 'float_speed', 1.0))
        
        for char_data in self.floating_chars:
            # Multiply the letter's unique randomized base speed by the global slider
            char_data['y'] -= (char_data['vel'] * speed_mult)
            
            fade_start_y = 650
            
            # v96 Fix: Safely cast the UI variable to a float to prevent string math crashes
            target_alt = float(getattr(self, 'fade_altitude', 290))
            fade_end_y = 650 - target_alt
            
            # Prevent division by zero if the math ever collapses
            if fade_start_y == fade_end_y: fade_end_y -= 1 
            
            # Fast-math fading logic
            raw_intensity = ((char_data['y'] - fade_end_y) / (fade_start_y - fade_end_y)) * 180
            intensity = max(0, min(180, int(raw_intensity)))
            
            if intensity > 0:
                # Fade the selected color toward black as it floats up
                factor = intensity / 180.0
                color = (int(base_b * factor), int(base_g * factor), int(base_r * factor), 255)
                
                cv2.putText(text_layer, char_data['char'], (int(char_data['x']), int(char_data['y'])), font, scale, color, thickness, cv2.LINE_AA)
                active_chars.append(char_data)
                
        # Clear out characters that floated off screen
        self.floating_chars = active_chars
        
        # Flip the text layer backwards so it reads correctly when the main app mirrors the video
        text_layer = cv2.flip(text_layer, 1)
        
        # Add the glowing text layer on top of the video frame
        final_frame = cv2.add(final_frame, text_layer)
        
        if final_frame is None or final_frame.shape[0] == 0:
            return frame
            
        return final_frame