import cv2
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from tkinter import colorchooser

class CustomFilter:
    def __init__(self):
        self.name = "Example: Tracking Wireframe"
        
        # ==============================================================================
        # 1. INITIALIZE DEDICATED TRACKING AI
        # ==============================================================================
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # refine_landmarks=True must be enabled to generate the Iris coordinates (468/473)
        self.face = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # ==============================================================================
        # 2. DEFAULT STATES & VISUAL STYLES
        # ==============================================================================
        self.joint_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=-1, circle_radius=4) 
        self.bone_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) 
        
        self.show_face = True
        self.show_hands = True
        
        # New GUI Variables
        self.poly_smooth = 0.04
        self.grid_density = 20
        self.grid_alpha = 0.4
        self.poly_alpha = 0.2
        self.poly_color_hex = "#00FF00" 
        self.eye_size = 15
        self.eye_color_hex = "#C8C8C8"
        
        # Grid cache system to prevent CPU load
        self.grid_cache = None
        self.last_density = -1
        self.last_h = 0
        self.last_w = 0
        
    def release(self):
        """Called automatically by v96 Node Editor to prevent memory leaks."""
        try:
            self.hands.close()
            self.face.close()
        except:
            pass

    def get_ui_schema(self):
        """v96 Architecture: Instructs the Node Editor to build the UI natively."""
        return [
            {"type": "switch", "name": "show_hands", "label": "Hand Bones", "default": True, "row": 0, "column": 0, "columnspan": 2},
            {"type": "switch", "name": "show_face", "label": "Face Polygon", "default": True, "row": 0, "column": 2, "columnspan": 2},
            
            {"type": "slider", "name": "poly_smooth", "label": "Poly Edges", "min": 0.001, "max": 0.08, "default": 0.04, "row": 1, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "grid_density", "label": "Grid Density", "min": 5, "max": 60, "default": 20, "row": 2, "column": 0, "columnspan": 4},
            {"type": "slider", "name": "grid_alpha", "label": "Grid Alpha", "min": 0.0, "max": 1.0, "default": 0.4, "row": 3, "column": 0, "columnspan": 4},
            
            {"type": "slider", "name": "poly_alpha", "label": "Fill Alpha", "min": 0.0, "max": 1.0, "default": 0.2, "row": 4, "column": 0, "columnspan": 3},
            {"type": "color", "name": "poly_color_hex", "label": "Fill Color", "default": "#00FF00", "row": 4, "column": 3, "columnspan": 1},
            
            {"type": "slider", "name": "eye_size", "label": "Eye Size", "min": 1, "max": 60, "default": 15, "row": 5, "column": 0, "columnspan": 3},
            {"type": "color", "name": "eye_color_hex", "label": "Eye Color", "default": "#C8C8C8", "row": 5, "column": 3, "columnspan": 1}
        ]

    def hex_to_bgr(self, hex_val):
        """Safely parses the Node Editor's hex color output back to OpenCV BGR."""
        hex_val = str(hex_val).lstrip('#')
        if len(hex_val) != 6: return (255, 255, 255)
        r, g, b = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    # ==========================================================
    # CACHE BUILDERS
    # ==========================================================
    def build_grid_cache(self, h, w):
        """Generates a full-screen grid once to save CPU cycles."""
        self.grid_cache = np.zeros((h, w, 3), dtype=np.uint8)
        spacing = max(2, int(self.grid_density))
        
        # Draw the light gray grid lines
        for x in range(0, w, spacing):
            cv2.line(self.grid_cache, (x, 0), (x, h), (200, 200, 200), 1)
        for y in range(0, h, spacing):
            cv2.line(self.grid_cache, (0, y), (w, y), (200, 200, 200), 1)
            
        # v96 Optimization: Pre-compute the binary mask so it isn't recalculated every frame
        gray_grid = cv2.cvtColor(self.grid_cache, cv2.COLOR_BGR2GRAY)
        _, self.grid_mask_cache = cv2.threshold(gray_grid, 1, 255, cv2.THRESH_BINARY)
            
        self.last_density = self.grid_density
        self.last_h = h
        self.last_w = w

    # ==========================================================
    # MAIN RENDER LOOP
    # ==========================================================
    def apply(self, frame, mask, hands_pointers, eyes):
        h, w = frame.shape[:2]
        
        # --- THREAD SAFETY: FREEZE GUI VARIABLES ---
        # Grabbing local copies so they don't mutate mid-render if the user drags a slider
        show_hands = getattr(self, 'show_hands', True)
        show_face = getattr(self, 'show_face', True)
        poly_smooth = getattr(self, 'poly_smooth', 0.04)
        poly_alpha = getattr(self, 'poly_alpha', 0.2)
        grid_alpha = getattr(self, 'grid_alpha', 0.4)
        grid_density = getattr(self, 'grid_density', 20)
        
        # Convert hex UI values back to BGR for OpenCV
        poly_color = self.hex_to_bgr(getattr(self, 'poly_color_hex', "#00FF00"))
        eye_color = self.hex_to_bgr(getattr(self, 'eye_color_hex', "#C8C8C8"))
        eye_size = getattr(self, 'eye_size', 15)

        # Keep our grid cache dynamically matched to the video resolution
        if self.grid_cache is None or self.last_density != grid_density or self.last_h != h or self.last_w != w:
            self.build_grid_cache(h, w)
            
        # v96 Optimization: Shrink first, then color convert. 
        # This reduces the CPU color-conversion workload from 921,600 pixels to just 57,600 pixels.
        small_bgr = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        ai_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        
        # ---------------------------------------------------------
        # A. DRAW FACE POLYGON & EYES (Z-Index: Bottom & Middle)
        # ---------------------------------------------------------
        if show_face:
            face_results = self.face.process(ai_frame)
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    
                    # --- 1. POLYGON MATH & FILL (Absolute Bottom Layer) ---
                    points = []
                    for lm in face_landmarks.landmark:
                        points.append([int(lm.x * w), int(lm.y * h)])
                    points = np.array(points, dtype=np.int32)
                    
                    hull = cv2.convexHull(points)
                    perimeter = cv2.arcLength(hull, True)
                    
                    epsilon = poly_smooth * perimeter 
                    simple_poly = cv2.approxPolyDP(hull, epsilon, True)
                    
                    poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [simple_poly], 255)
                    
                    # A. Apply Solid Color Fill (Fast C++ Blending)
                    if poly_alpha > 0.01:
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [simple_poly], poly_color)
                        # Since 'overlay' only differs from 'frame' inside the polygon, 
                        # addWeighted creates a perfectly localized alpha blend instantly!
                        cv2.addWeighted(overlay, poly_alpha, frame, 1.0 - poly_alpha, 0, dst=frame)
                    
                    # B. Apply Light Gray Grid Line Fill (Fast C++ Blending)
                    if grid_alpha > 0.01:
                        grid_overlay = frame.copy()
                        
                        # Use the pre-computed static mask from the cache
                        active_lines = cv2.bitwise_and(self.grid_mask_cache, poly_mask)
                        grid_overlay[active_lines > 0] = (200, 200, 200)
                        
                        cv2.addWeighted(grid_overlay, grid_alpha, frame, 1.0 - grid_alpha, 0, dst=frame)
                    
                    # C. Draw Outer Boundary Line
                    cv2.polylines(frame, [simple_poly], isClosed=True, color=poly_color, thickness=2)

                    # --- 2. EYE TRACKING (Middle Layer) ---
                    # 468 is the right iris center, 473 is the left iris center
                    rx, ry = int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h)
                    lx, ly = int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h)
                    
                    e_w = max(1, int(eye_size * 2.0))
                    e_h = max(1, int(eye_size * 0.8))
                    
                    cv2.ellipse(frame, (rx, ry), (e_w, e_h), 0, 0, 360, eye_color, -1)
                    cv2.ellipse(frame, (lx, ly), (e_w, e_h), 0, 0, 360, eye_color, -1)

        # ---------------------------------------------------------
        # B. DRAW HAND BONES (Z-Index: Top Layer)
        # ---------------------------------------------------------
        if show_hands:
            hand_results = self.hands.process(ai_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.joint_spec,
                        connection_drawing_spec=self.bone_spec
                    )

        return frame