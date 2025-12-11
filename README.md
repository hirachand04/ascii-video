# 📺 HD Real-Time ASCII Video Converter

Turn your webcam feed into a stunning, high-definition ASCII art stream in real-time.

![ASCII Art Preview](https://img.shields.io/badge/Quality-HD%20720p-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Win%20|%20Mac%20|%20Linux%20|%20Android-blue)
![Python](https://img.shields.io/badge/Python-3.7+-yellow)

**New in v2.0:**
- 🖥️ **Full HD Resolution**: Renders at native 1280x720 resolution for crisp, pixel-perfect characters.
- 📺 **Fullscreen Support**: Automatically adapts to any window size—maximize it for the best experience!
- 🎨 **6 Rendering Modes**: From "High Contrast" visibility to "Matrix" style.
- 🚀 **High Performance**: Optimized to run smoothly at 30+ FPS.

---

## ⚡ Quick Start

### Windows / macOS / Linux

1. **Clone the repo**
   ```bash
   git clone https://github.com/hirachand04/ascii-video.git
   cd ascii-video
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run it!**
   ```bash
   py -3.13 main.py
   ```
   **Or**
   ```bash
   python main.py
   ```
   *Click the window and press `F11` or maximize for the full immersive experience!*

## 🎮 Controls

**IMPORTANT**: Click on the "ASCII Video Preview" window to ensure it has focus before using keys.

| Key | Action |
|-----|--------|
| `m` | **Cycle Modes** (Standard, High Contrast, Matrix, Blocks, etc.) |
| `+` / `-` | **Adjust Density** (Increase/Decrease number of characters) |
| `c` | **Cycle Colors** (Green, Cyan, White, Magenta, Orange) |
| `k` | **Toggle Color Mode** (Monochrome vs. Real-Color) |
| `i` | **Invert** (Dark Mode / Light Mode) |
| `o` | **Toggle Mirror** (Show/Hide original webcam feed) |
| `r` | **Reset** to default settings |
| `q` | **Quit** |

---

## 🛠️ Configuration

### Rendering Modes
- **Simple HD** (Default): Best balance of visibility and detail.
- **High Contrast**: Uses block characters for clear object recognition.
- **Matrix**: Uses classic matrix-style characters.
- **Blocks**: Smooth gradients using full-block characters.

### Performance Tips
- If FPS is low, press `-` to reduce the character count (this increases character size).
- Maximize the window—the app automatically fills whatever space you give it!

---

## 📄 License
MIT License. Feel free to use and modify!



