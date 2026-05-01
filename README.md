# PixelBoost

PixelBoost is a polished Streamlit web app for AI-powered image upscaling. Users can upload a low-resolution image, enhance it with a pretrained super-resolution model, compare before and after results, and download the improved output.

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

## App Workflow

1. Open the Streamlit app.
2. Upload a supported image.
3. Review the original preview and metadata.
4. Choose `2x` or `4x` upscale.
5. Click `Upscale Image`.
6. Compare the original and enhanced outputs.
7. Download the processed image.

## Notes

- Real-ESRGAN usually provides stronger visual quality, but it has a heavier dependency footprint.
- The app caches the selected upscaler so the model does not reload for every request.
- Transparent PNG images are supported; alpha is restored after RGB upscaling.
- Large `4x` images on CPU automatically switch to a faster progressive OpenCV path for a smoother demo experience.

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
