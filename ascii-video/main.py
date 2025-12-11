"""
Real-Time ASCII Video Converter
===============================
This application captures video from your webcam and converts it to ASCII art
in real-time, displaying the result in a live preview window.

Usage:
    python main.py

Controls:
    q - Quit the application
    i - Toggle inverted colors
    c - Cycle through character sets
    + - Increase ASCII width
    - - Decrease ASCII width
    r - Reset to default settings
"""

import cv2
import sys
import time
import numpy as np
from ascii_converter import frame_to_ascii_image, get_optimal_dimensions


class ASCIIVideoConverter:
    """Main class for real-time ASCII video conversion."""
    
    def __init__(self):
        self.cap = None
        self.ascii_width = 200  # Higher resolution - more columns = more detail
        self.min_width = 100
        self.max_width = 350
        self.mode = "simple_hd"  # ASCII characters for terminal-like look
        self.mode_options = ["standard", "extended", "highcontrast", "simple_hd", "blocks", "binary"]
        self.mode_index = 3
        self.invert = False
        self.use_color = False  # Color ASCII mode
        self.target_fps = 30
        self.show_original = True
        self.running = False
        
        # Color presets
        self.color_presets = [
            (0, 255, 0),    # Green (Matrix)
            (0, 255, 255),  # Yellow
            (255, 255, 255), # White
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
        ]
        self.color_index = 0
        
        # Display settings
        self.window_name = "ASCII Video Preview"
        self.original_window = "Original Feed"
        self.bg_color = (0, 0, 0)        # Black background
        self.fg_color = self.color_presets[0]  # Green text
        
        # Performance tracking
        self.frame_times = []
        self.fps_display = 0
        
        # Dynamic window size (updated each frame)
        self.output_width = 1280
        self.output_height = 720
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize the camera with graceful error handling.
        
        Parameters:
        -----------
        camera_index : int
            Camera device index (0 for default webcam)
        
        Returns:
        --------
        bool
            True if camera initialized successfully, False otherwise
        """
        print(f"Attempting to access camera {camera_index}...")
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print("\n❌ ERROR: Could not open camera.")
                print("Possible reasons:")
                print("  1. No camera connected")
                print("  2. Camera is in use by another application")
                print("  3. Camera permissions denied")
                print("\nTroubleshooting:")
                print("  - Check if your camera is properly connected")
                print("  - Close other applications using the camera")
                print("  - Grant camera permissions to this application")
                return False
            
            # Try to read a test frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("\n❌ ERROR: Camera opened but couldn't capture frame.")
                print("The camera may be blocked or malfunctioning.")
                self.cap.release()
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"\n✅ Camera initialized successfully!")
            print(f"   Resolution: {actual_width}x{actual_height}")
            print(f"   Target FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: Failed to access camera: {e}")
            print("Please check your camera permissions and try again.")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a video frame to ASCII art image.
        
        Parameters:
        -----------
        frame : np.ndarray
            Input BGR frame from camera
        
        Returns:
        --------
        np.ndarray
            Image with rendered ASCII art
        """
        # Flip frame horizontally to fix mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to ASCII image with selected mode and color
        # Output adapts to current window size
        ascii_image = frame_to_ascii_image(
            frame,
            width=self.ascii_width,
            fg_color=self.fg_color,
            bg_color=self.bg_color,
            invert=self.invert,
            mode=self.mode,
            use_color=self.use_color,
            output_width=self.output_width,
            output_height=self.output_height
        )
        
        return ascii_image
    
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frame times
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate FPS from frame times
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps_display = (len(self.frame_times) - 1) / time_diff
    
    def add_info_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add information overlay to the image."""
        h, w = image.shape[:2]
        
        # Create info text
        info_lines = [
            f"FPS: {self.fps_display:.1f}",
            f"Width: {self.ascii_width}",
            f"Mode: {self.mode}",
            f"Color: {'RGB' if self.use_color else 'mono'}",
            f"Inverted: {self.invert}"
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (200, 105), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 22
        for line in info_lines:
            cv2.putText(image, line, (10, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18
        
        return image
    
    def handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Returns:
        --------
        bool
            True to continue running, False to quit
        """
        if key == ord('q') or key == ord('Q'):
            return False
        elif key == ord('i') or key == ord('I'):
            self.invert = not self.invert
            print(f"Inverted: {self.invert}")
        elif key == ord('m') or key == ord('M'):
            # Cycle through modes: standard, extended, binary, blocks
            self.mode_index = (self.mode_index + 1) % len(self.mode_options)
            self.mode = self.mode_options[self.mode_index]
            print(f"Mode: {self.mode}")
        elif key == ord('k') or key == ord('K'):
            # Toggle color mode
            self.use_color = not self.use_color
            print(f"Color mode: {'RGB' if self.use_color else 'mono'}")
        elif key == ord('c') or key == ord('C'):
            # Cycle color presets (only in mono mode)
            if not self.use_color:
                self.color_index = (self.color_index + 1) % len(self.color_presets)
                self.fg_color = self.color_presets[self.color_index]
                colors = ["Green", "Yellow", "White", "Magenta", "Orange"]
                print(f"Color: {colors[self.color_index]}")
        elif key == ord('+') or key == ord('='):
            self.ascii_width = min(self.max_width, self.ascii_width + 20)
            print(f"ASCII Width: {self.ascii_width}")
        elif key == ord('-') or key == ord('_'):
            self.ascii_width = max(self.min_width, self.ascii_width - 20)
            print(f"ASCII Width: {self.ascii_width}")
        elif key == ord('r') or key == ord('R'):
            self.reset_settings()
            print("Settings reset to defaults")
        elif key == ord('o') or key == ord('O'):
            self.show_original = not self.show_original
            if not self.show_original:
                cv2.destroyWindow(self.original_window)
            print(f"Show original: {self.show_original}")
        
        return True
    
    def reset_settings(self):
        """Reset all settings to defaults."""
        self.ascii_width = 200
        self.mode = "simple_hd"
        self.mode_index = 3
        self.invert = False
        self.use_color = False
        self.color_index = 0
        self.fg_color = self.color_presets[0]
    
    def run(self):
        """Main loop for the ASCII video converter."""
        print("\n" + "="*50)
        print("  Real-Time ASCII Video Converter")
        print("="*50)
        
        # Initialize camera
        if not self.initialize_camera():
            print("\nExiting due to camera initialization failure.")
            return
        
        print("\n Starting ASCII video preview...")
        print("\n Click on the 'ASCII Video Preview' window to use keyboard controls!")
        print("\n ASCII Camera Renderer v1.0")
        print("\n Created by: Hirachand ")
        print(" GitHub: https://github.com/hirachand04")

        print("\nControls (window must be focused):")
        print("  q - Quit")
        print("  m - Cycle modes (standard/extended/highcontrast/simple_hd/blocks/binary)")
        print("  k - Toggle color mode (mono/RGB)")
        print("  c - Cycle mono colors (green/yellow/white/magenta/orange)")
        print("  i - Invert brightness")
        print("  +/- - Adjust resolution (more/fewer characters)")
        print("  o - Toggle original feed")
        print("  r - Reset settings")
        print("\n")
        
        self.running = True
        
        # Create resizable window for fullscreen support
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Warning: Failed to capture frame, retrying...")
                    continue
                
                # Update FPS
                self.update_fps()
                
                # Process frame to ASCII
                ascii_image = self.process_frame(frame)
                
                # Add info overlay
                ascii_image = self.add_info_overlay(ascii_image)
                
                # Display ASCII preview
                cv2.imshow(self.window_name, ascii_image)
                
                # Get current window size and update for next frame
                # This enables adaptive fullscreen rendering
                try:
                    rect = cv2.getWindowImageRect(self.window_name)
                    if rect[2] > 0 and rect[3] > 0:  # Valid dimensions
                        self.output_width = rect[2]
                        self.output_height = rect[3]
                except:
                    pass  # Keep previous size if detection fails
                
                # Optionally display original feed - compact window
                if self.show_original:
                    # Flip and resize original for compact display
                    flipped = cv2.flip(frame, 1)
                    # Smaller preview window (320x240)
                    original_display = cv2.resize(flipped, (320, 240))
                    cv2.imshow(self.original_window, original_display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_keypress(key):
                        break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete. Goodbye!")


def main():
    """Entry point for the application."""
    converter = ASCIIVideoConverter()
    converter.run()


if __name__ == "__main__":
    main()
