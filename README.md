# PixelBoost

PixelBoost is a polished Streamlit web app for AI-powered image upscaling. Users can upload a low-resolution image, enhance it with a pretrained super-resolution model, compare before and after results, and download the improved output.

## Why PixelBoost

PixelBoost was built as a resume-ready AI project that feels like a real product instead of a basic script. It combines:

- A usable web interface built with Streamlit
- High-quality super-resolution through Real-ESRGAN
- A reliable OpenCV fallback path for practical demos
- Clear before/after proof, metadata, and downloadable results

## Highlights

- `High Quality Mode`: Real-ESRGAN for sharper enhancement
- `Reliable Demo Path`: OpenCV fallback when heavier dependencies are unavailable
- `Visual Comparison`: Before/after slider for easy result validation
- `Insightful UI`: Metadata, estimated output size, processing time, and result summary
- `Portfolio Friendly`: Modular code, setup docs, troubleshooting, and resume bullets

## Quick Start

If the project is already configured locally, the shortest launch path is:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then visit:

```text
http://127.0.0.1:8501
```

## Features

- Single image upload with support for `JPG`, `JPEG`, `PNG`, and `WEBP`
- AI-powered upscaling using pretrained `Real-ESRGAN`
- Fallback support through pretrained OpenCV DNN super-resolution models
- Original and enhanced image previews
- Before/after comparison slider when `streamlit-image-comparison` is installed
- Downloadable output image
- Input and output dimension display
- Progress and status messaging during processing
- Clear error handling for invalid files, missing models, or processing failures

## Tech Stack

- Python
- Streamlit
- Pillow
- NumPy
- OpenCV
- Real-ESRGAN / ESRGAN-style pretrained models
- Optional: `streamlit-image-comparison`

## Project Structure

```text
PixelBoost/
|-- app.py
|-- requirements.txt
|-- README.md
|-- models/
|   |-- opencv/
|   `-- realesrgan/
|-- pixelboost/
|   |-- __init__.py
|   |-- config.py
|   |-- image_utils.py
|   `-- super_resolution.py
`-- scripts/
    `-- download_opencv_models.ps1
```

## Architecture Overview

- `app.py` handles the Streamlit interface, layout, user interactions, and result rendering.
- `pixelboost/image_utils.py` handles image loading, metadata extraction, conversion helpers, and download output generation.
- `pixelboost/super_resolution.py` manages backend selection between Real-ESRGAN and the OpenCV fallback path.
- `pixelboost/config.py` centralizes model paths, supported formats, and runtime thresholds.

## Setup

1. Create and activate a virtual environment.
2. Install core dependencies:

```bash
pip install -r requirements.txt
```

Windows project-local environment example:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. Optional: install the heavier Real-ESRGAN stack for the preferred backend:

```bash
pip install realesrgan basicsr torch torchvision
```

4. Create the model folders:

```bash
mkdir models
mkdir models\realesrgan
mkdir models\opencv
```

## Model Weights

PixelBoost prefers Real-ESRGAN and falls back to lighter pretrained OpenCV models when Real-ESRGAN is unavailable.

### Preferred: Real-ESRGAN Weights

Download these official weights from the Real-ESRGAN project releases and place them in `models/realesrgan/`:

- `RealESRGAN_x2plus.pth`
  Source: [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- `RealESRGAN_x4plus.pth`
  Source: [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

Expected layout:

```text
models/
`-- realesrgan/
    |-- RealESRGAN_x2plus.pth
    `-- RealESRGAN_x4plus.pth
```

### Fallback: OpenCV DNN Super-Resolution Models

If you do not want the heavier Real-ESRGAN setup, download these pretrained TensorFlow `.pb` files and place them in `models/opencv/`:

- `FSRCNN_x2.pb`
- `EDSR_x4.pb`

You can also use the included PowerShell helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_opencv_models.ps1
```

OpenCV documents the supported `dnn_superres` workflow here:
[OpenCV Super Resolution Tutorial](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html)

Expected layout:

```text
models/
`-- opencv/
    |-- FSRCNN_x2.pb
    `-- EDSR_x4.pb
```

## Run Locally

```bash
streamlit run app.py
```

If you are using the project-local virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

## App Workflow

1. Open the Streamlit app.
2. Upload a supported image.
3. Review the original preview and metadata.
4. Choose `2x` or `4x` upscale.
5. Click `Upscale Image`.
6. Compare the original and enhanced outputs.
7. Download the processed image.

## Choosing 2x vs 4x

- Use `2x` when you want a faster enhancement pass and a lighter demo flow.
- Use `4x` when the image is very small and you want a stronger jump in output resolution.
- For slower machines, start with `2x` first and move to `4x` only when you need more detail.

## Demo Flow

Use this sequence for a clean college or interview demo:

1. Open the app and explain that PixelBoost is an AI image upscaling tool.
2. Upload a visibly compressed or low-resolution image.
3. Show the input metadata and estimated output resolution.
4. Run `2x` first for a quick proof, then `4x` for a stronger enhancement pass.
5. Use the comparison slider to point out sharper edges, clearer textures, and better visual detail.
6. Highlight the processing time, output summary, and download flow.
7. Mention that Real-ESRGAN is the preferred quality backend and OpenCV is kept as a practical fallback.

## Notes

- Real-ESRGAN usually provides stronger visual quality, but it has a heavier dependency footprint.
- The app caches the selected upscaler so the model does not reload for every request.
- Transparent PNG images are supported; alpha is restored after RGB upscaling.
- Large `4x` images on CPU automatically switch to a faster progressive OpenCV path for a smoother demo experience.

## Project Strengths

- Combines AI model usage with a visible end-user product workflow.
- Gives both technical and visual proof through metrics, metadata, and comparison UI.
- Stays practical for demos because it can fall back gracefully when a heavy backend is unavailable.
- Uses modular Python files that are easy to explain during code walkthroughs.

## Troubleshooting

- If the app says a model file is missing, confirm the `.pth` or `.pb` file is placed in the exact folder shown in the README.
- If `4x` feels slow on a laptop CPU, try `2x` first or test with a smaller input image.
- If the comparison slider does not appear, make sure `streamlit-image-comparison` is installed from `requirements.txt`.
- If Real-ESRGAN dependencies are heavy for your system, keep using the OpenCV fallback path for demos.
- If the app does not start from the project virtual environment, run `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Resume-Friendly Summary

PixelBoost demonstrates:

- AI model integration using pretrained image super-resolution networks
- Streamlit app development for an end-to-end product workflow
- Image preprocessing and postprocessing with Pillow and OpenCV
- Error handling, caching, and deployment-ready project structure

## Demo and Resume Tips

Use these points while presenting PixelBoost:

- Explain that the app prefers Real-ESRGAN when available, but includes a practical OpenCV fallback so demos stay reliable on CPU systems.
- Show one compressed portrait or product image and compare sharpness around edges, facial details, and text-like regions.
- Mention that the app surfaces metadata, estimated output size, processing time, and before/after proof to make the result easier to evaluate.
- Highlight that the codebase is modular, with separate helpers for UI, image handling, and super-resolution logic.

Short interview summary:

- PixelBoost is an end-to-end AI image upscaling application built for real usage, not just model experimentation.
- It integrates pretrained super-resolution models into a web product workflow with upload, preview, enhancement, comparison, and export.
- It also handles practical engineering concerns such as fallbacks, processing feedback, metadata visibility, and demo reliability.

Sample resume bullets:

- Built `PixelBoost`, an AI image upscaling web app using Python, Streamlit, and pretrained super-resolution models.
- Implemented upload, enhancement, comparison, and export workflows with CPU-friendly fallback logic for reliable demos.
- Improved usability through metadata panels, runtime feedback, performance safeguards, and download-ready outputs.
