import cv2
import numpy as np
import mediapipe as mp
import random

# This effect makes the lightning happen (as promised :)
# - It's a good example of fast tracking, while also showing how to maintain speed on slower GPUs
# - Tracking resolution should be reduced to 320x180, which makes it very responsive for mediapipe.
#   But if you raise it much, it gets slower (CPU-bound)
# - If you want to track way faster, onnxruntime-gpu is built-in, so you can write a script
#   that drops an .onnx model file right next to your .py script, loads it into the
#   CustomFilter class, and run on GPU. It's wild. I will make an example of that for next release.
#   
# - Also you can see how to set up your mini-gui controls for sliders and a switches. You can
#   just add another switch by copy-paste, and make it do whatever you want.
# 
class CustomFilter:
    def __init__(self):
        self.name = "Effect: Lightning Fingers"
        
        # ==============================================================================
        # 1. INITIALIZE DEDICATED TRACKING AI
        # ==============================================================================
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0, 
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # ==============================================================================
        # 2. DEFAULT STATES & VISUAL STYLES
        # ==============================================================================
        # Dynamic GUI Variables
        self.bolt_thickness = 2
        self.glow_size = 6
        self.bolt_branches = 2
        self.core_color_hex = "#FFFFFF"
        self.glow_color_hex = "#00E5FF"
        
        # Mode Toggle State: False = Two Finger (Left), True = Five Finger (Right)
        self.mode_five_finger = False

    def release(self):
        try:
            self.hands.close()
        except:
            pass

    def get_ui_schema(self):
        """v96 Architecture UI Layout Definition"""
        return [
            # Switch Component
            {"type": "switch", "name": "mode_five_finger", "label_left": "Two finger", "label_right": "Five finger", "default": False, "row": 0, "column": 0, "columnspan": 4},
            
            {"type": "slider", "name": "bolt_thickness", "label": "Core Width", "min": 1, "max": 5, "default": 2, "row": 1, "column": 0, "columnspan": 2},
            {"type": "slider", "name": "glow_size", "label": "Glow Radius", "min": 0, "max": 15, "default": 6, "row": 1, "column": 2, "columnspan": 2},
            
            {"type": "slider", "name": "bolt_branches", "label": "Branches/Finger", "min": 1, "max": 4, "default": 2, "row": 2, "column": 0, "columnspan": 4},
            
            {"type": "color", "name": "core_color_hex", "label": "Core Color", "default": "#FFFFFF", "row": 3, "column": 0, "columnspan": 2},
            {"type": "color", "name": "glow_color_hex", "label": "Glow Plasma", "default": "#00E5FF", "row": 3, "column": 2, "columnspan": 2}
        ]

    def hex_to_bgr(self, hex_val):
        hex_val = str(hex_val).lstrip('#')
        if len(hex_val) != 6: return (255, 255, 255)
        r, g, b = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def draw_plasma_arc(self, img, pt1, pt2, core_color, glow_color, thickness, glow_radius):
        points = [pt1]
        cur_pt = np.array(pt1, dtype=np.float32)
        target_pt = np.array(pt2, dtype=np.float32)
        
        vec = target_pt - cur_pt
        dist = np.linalg.norm(vec)
        if dist < 10: 
            return
            
        num_segments = 6
        segment_vec = vec / num_segments
        
        perp = np.array([-vec[1], vec[0]], dtype=np.float32)
        norm_perp = np.linalg.norm(perp)
        if norm_perp > 0:
            perp = perp / norm_perp
            
        for i in range(1, num_segments):
            mid_pt = cur_pt + segment_vec * i
            displace_magnitude = random.uniform(-dist * 0.08, dist * 0.08)
            displaced_pt = mid_pt + perp * displace_magnitude
            points.append((int(displaced_pt[0]), int(displaced_pt[1])))
            
        points.append(pt2)
        
        if glow_radius > 0:
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i+1], glow_color, int(thickness + glow_radius))
                
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], core_color, int(thickness))

    # ==========================================================
    # MAIN RENDER LOOP
    # ==========================================================
    def apply(self, frame, mask, hands_pointers, eyes):
        h, w = frame.shape[:2]
        
        # --- THREAD SAFETY STATIC SNAPSHOTS ---
        mode_five_finger = getattr(self, 'mode_five_finger', False)
        bolt_thickness = getattr(self, 'bolt_thickness', 2)
        glow_size = getattr(self, 'glow_size', 6)
        bolt_branches = int(getattr(self, 'bolt_branches', 2))
        
        core_color = self.hex_to_bgr(getattr(self, 'core_color_hex', "#FFFFFF"))
        glow_color = self.hex_to_bgr(getattr(self, 'glow_color_hex', "#00E5FF"))

        # Performance downsampling pipeline (320x180 workload restriction)
        small_bgr = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        ai_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        
        hand_results = self.hands.process(ai_frame)
        
        # Only process the lightning if EXACTLY two hands are visible on camera
        if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) >= 2:
            
            # Map the two detected hands
            hand_a = hand_results.multi_hand_landmarks[0]
            hand_b = hand_results.multi_hand_landmarks[1]

            # Determine the circuit points based on UI Switch
            if mode_five_finger:
                # Target all five matching pairs
                target_fingertips = [4, 8, 12, 16, 20]
            else:
                # Target only the index pointer fingers (ID 8)
                target_fingertips = [8]

            # Draw arcs between the corresponding joints
            for tip_id in target_fingertips:
                lm_a = hand_a.landmark[tip_id]
                lm_b = hand_b.landmark[tip_id]
                
                # Convert normalized coordinates to absolute pixels
                start_x, start_y = int(lm_a.x * w), int(lm_a.y * h)
                target_x, target_y = int(lm_b.x * w), int(lm_b.y * h)
                
                # Generate the jagged bolt layers
                for _ in range(bolt_branches):
                    self.draw_plasma_arc(
                        frame, 
                        (start_x, start_y), 
                        (target_x, target_y), 
                        core_color, 
                        glow_color, 
                        bolt_thickness, 
                        glow_size
                    )

        return frame