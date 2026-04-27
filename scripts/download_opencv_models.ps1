$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$opencvDir = Join-Path $projectRoot "models\\opencv"

New-Item -ItemType Directory -Force -Path $opencvDir | Out-Null

$downloads = @(
    @{
        Url = "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb"
        Output = Join-Path $opencvDir "FSRCNN_x2.pb"
    },
    @{
        Url = "https://raw.githubusercontent.com/metinakkin/EDSR_Performance/main/EDSR_x4.pb"
        Output = Join-Path $opencvDir "EDSR_x4.pb"
    }
)

foreach ($item in $downloads) {
    Write-Host "Downloading $($item.Url)"
    Invoke-WebRequest -Uri $item.Url -OutFile $item.Output
}

Write-Host "OpenCV fallback models downloaded to $opencvDir"
