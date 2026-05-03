$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $pythonExe -c "from PIL import Image; from pixelboost.super_resolution import upscale_image; img=Image.new('RGB',(24,24),(120,80,200)); out2, backend2=upscale_image(img,2); out4, backend4=upscale_image(img,4); print('2x:', backend2, out2.size); print('4x:', backend4, out4.size)"
