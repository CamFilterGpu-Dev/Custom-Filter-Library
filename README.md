# CamFilterGpu - Example Filter Library

Official Python script examples and templates for creating custom video effects and filters in **CamFilterGpu**. 

CamFilterGpu is a fast live video *super-enhancer* with foreground-background separation and independent color grading, a unique effects generator, and built-in GPU acceleration with AI segmentation/tracking. CamFilterGpu was designed from scratch to allow you to rapidly build video pipelines and inject your own filtering scripts 'on-the-fly'. There is a built-in Node Pipeline Editor (see the video demos) where you can select a Python script and load it for instant viewing and tweaking of your custom special effects!
CamFilterGpu is designed to work seamlessly and zero-lag with OBS (via Spout2), Zoom, Discord, and other related systems.

* 🎬 **Video Demos & Tutorials:** [youtube.com/@camfiltergpu](https://youtube.com/@camfiltergpu)
* 🚀 **Download the App:** Visit [Perfunct.com](https://perfunct.com) *(Note: The main application is proprietary, not open source, but is very inexpensive, like cup-of-coffee inexpensive, 14 day free trial)*

These initial scripts demonstrate how to hook into the GPU-accelerated virtual camera pipeline, allowing you to manipulate live video feeds using standard Python and OpenCV operations. Many other powerful libraries are already built-in, such as `numpy`, `onnxruntime`, and more.

## 📦 What's Included

This repository contains 5 baseline example scripts designed to help you understand the filter architecture and get a quick start:
* **intro_example_newsprint.py** - *Transforms the video into 'newsprint-like' halftone*
* **example_01_basic_noise.py** - *Adds moving 'noise' to the video, both black and white or color, with an inverter switch*
* **example_02_risingtype.py)** - *Enables live typing in the video window. When Enter is pressed, the letters 'float' upwards*
* **example_03_asciimask.py** - *Transforms the subject profile into ASCII text, with control of letters, size, transparency, and font/color*
* **example_04_handfacetrack-fullgui.py** - *Performs face and hand tracking with a polygon mask, 'eyeballs', and hand 'skeletons'*
* **example_05_sharpenAlloptim.py** - *Performs ultra-fast and extreme sharpening, from subtle to extreme, with text too, and inverter*
* **example_06_lightning_fingers** - *Makes blue lightning bolts appear between fingers on each hand (coming soon)*

## 🚀 How to Use

1. Download the `.py` script you want to use.
2. Place the script into your designated CamFilterGpu Custom Filters directory.
3. Launch (or restart) CamFilterGpu. (Get it from [Perfunct.com](https://perfunct.com), 14 day free trial)
4. Select the filter from Node Editor Window (see demo videos), apply it to your live virtual camera feed. That's it!

## 🛠️ Development & Customization

These scripts are built to be modified! You can open any of these `.py` files in your preferred text editor or IDE. Because CamFilterGpu handles the heavy lifting of the GPU acceleration and the virtual camera routing, you can focus purely on writing the OpenCV or AI/tracking logic for your specific visual effect.
The creative possibilites are endless.

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to download, modify, distribute, and use these example scripts in your own personal or commercial projects.
