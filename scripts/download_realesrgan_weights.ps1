$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$weightsDir = Join-Path $projectRoot "models\\realesrgan"

New-Item -ItemType Directory -Force -Path $weightsDir | Out-Null

$downloads = @(
    @{
        Url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        Output = Join-Path $weightsDir "RealESRGAN_x2plus.pth"
    },
    @{
        Url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        Output = Join-Path $weightsDir "RealESRGAN_x4plus.pth"
    }
)

foreach ($item in $downloads) {
    Write-Host "Downloading $($item.Url)"
    Invoke-WebRequest -Uri $item.Url -OutFile $item.Output
}

Write-Host "Real-ESRGAN weights downloaded to $weightsDir"
