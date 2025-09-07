# Face Anonymizer

![License](https://img.shields.io/badge/license-MIT-green)
![Conventional Commits](https://img.shields.io/badge/commits-conventional-brightgreen)
![Python](https://img.shields.io/badge/language-Python-blue)
![Release](https://img.shields.io/badge/release-v0.1.0-orange)

## Overview

Face Anonymizer is a privacy‑focused tool that automatically detects and anonymizes faces in images, videos, or webcam streams. It supports multiple obfuscation modes such as **blur**, **pixelate**, and **emoji overlay**. The project is designed as a portfolio‑ready demo with a clean CLI, documentation, and tests.

## Repository Structure

```
face-anonymizer/
├─ src/              # Source code (main CLI)
├─ test/             # Tests and examples
├─ docs/             # Documentation
│  └─ assets/        # Images, diagrams
├─ models/           # Model files (not tracked)
├─ data/             # Local datasets (not tracked)
├─ notebooks/        # Jupyter notebooks (optional)
├─ build/            # Output files (not tracked)
├─ configs/          # Config files (YAML)
├─ archive/          # Legacy code
├─ requirements.txt  # Minimal dependencies
├─ LICENSE           # MIT License
└─ README.md         # Project description
```

## Getting Started

### Prerequisites

* Python 3.10+
* Install dependencies:

```powershell
pip install -r requirements.txt
```

### Usage

#### Anonymize an Image

```powershell
python .\src\main.py --input .\data\your_image.jpg --method blur --output .\build\out.jpg
```

#### Anonymize a Video

```powershell
python .\src\main.py --input .\data\your_video.mp4 --method pixelate --output .\build\out_video.mp4
```

#### Anonymize from Webcam

```powershell
python .\src\main.py --webcam 0 --method blur --draw --output .\build\out_cam.mp4
```

#### Use Emoji Overlay

```powershell
python .\src\main.py --input .\data\your_image.jpg --method emoji --emoji .\data\emoji.png --output .\build\emoji_image.png
```

### CLI Options

* `--input / -i`: Path to image or video.
* `--webcam / -w`: Webcam index (default 0).
* `--method / -m`: Anonymization method (`blur`, `pixelate`, `emoji`).
* `--emoji`: Path to PNG emoji (required for `--method emoji`).
* `--draw`: Draw bounding boxes around detected faces.
* `--output / -o`: Output path for anonymized media.
* `--config / -c`: YAML config file for detector settings.

## Dataset & Models

* Place your own media in `data/`. See [data/README.md](data/README.md).
* Place pretrained models or cascade files in `models/` (not tracked). Add instructions in `models/README.md`.

## Features

* Real‑time face detection using MediaPipe.
* Multiple anonymization modes (blur, pixelate, emoji overlay).
* Simple CLI with PowerShell examples.
* Modular codebase with configs and archive for legacy code.

## What I Learned

* Practical integration of MediaPipe FaceDetection with OpenCV.
* Building a CLI‑based tool with relative paths and configs.
* Managing datasets/models in a portfolio‑ready structure.
* Using conventional commits and clear repository scaffolding.

## Roadmap

* [ ] Add test suite in `test/`.
* [ ] Expand anonymization methods (mosaic, cartoon).
* [ ] Dockerfile for containerized usage.
* [ ] Optional GPU acceleration.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
