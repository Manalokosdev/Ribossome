# Package Ribossome for distribution
# Creates a standalone folder with all necessary files

$ErrorActionPreference = "Stop"

Write-Host "Building release version..." -ForegroundColor Cyan
cargo build --release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

$version = Get-Date -Format "yyyy-MM-dd"
$packageName = "Ribossome_$version"
$packageDir = ".\dist\$packageName"

Write-Host "Creating package directory: $packageDir" -ForegroundColor Cyan
Remove-Item -Path ".\dist\$packageName" -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $packageDir -Force | Out-Null

Write-Host "Copying executable..." -ForegroundColor Cyan
Copy-Item ".\target\release\ribossome.exe" -Destination $packageDir

Write-Host "Copying required folders..." -ForegroundColor Cyan
Copy-Item ".\config" -Destination "$packageDir\config" -Recurse
Copy-Item ".\shaders" -Destination "$packageDir\shaders" -Recurse
Copy-Item ".\maps" -Destination "$packageDir\maps" -Recurse

Write-Host "Copying documentation..." -ForegroundColor Cyan
Copy-Item ".\docs\README.md" -Destination "$packageDir\README.md"
Copy-Item ".\docs\LICENSE" -Destination "$packageDir\LICENSE" -ErrorAction SilentlyContinue

Write-Host "Creating empty directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path "$packageDir\recordings" -Force | Out-Null
New-Item -ItemType Directory -Path "$packageDir\screenshots" -Force | Out-Null
New-Item -ItemType Directory -Path "$packageDir\clips" -Force | Out-Null

Write-Host "Creating README_FIRST.txt..." -ForegroundColor Cyan
@"
RIBOSSOME - Artificial Life Simulator
=====================================

IMPORTANT - Windows SmartScreen Warning:
Windows will show a "Windows protected your PC" warning because this app is not
code-signed (certificates cost $$). This is normal for open-source software.

To run:
1. Click "More info" on the blue warning screen
2. Click "Run anyway" at the bottom
3. This only happens the first time

To run the simulator:
1. Double-click ribossome.exe (see SmartScreen note above)
2. The first run may take 10-60 seconds to compile shaders
3. Subsequent runs will be instant

Requirements:
- Windows 10/11 with a GPU that supports Vulkan or DirectX 12
- ffmpeg in PATH (optional - only needed for video/GIF recording)

Controls:
- WASD: Pan camera
- Mouse wheel: Zoom
- Right-drag: Pan
- Left-click: Select agent
- Space: Toggle UI
- F: Follow selected agent
- R: Reset camera

Folders:
- config/: Configuration files (edit to customize simulation)
- shaders/: GPU compute shaders
- maps/: Environment maps
- recordings/: Video/GIF recordings (created when you record)
- screenshots/: Screenshots (created when you take screenshots)
- clips/: Snapshot files (created when you save snapshots)

For more information, see README.md or visit:
https://github.com/Manalokosdev/Ribossome

"@ | Out-File -FilePath "$packageDir\README_FIRST.txt" -Encoding UTF8

Write-Host "Compressing to ZIP..." -ForegroundColor Cyan
$zipPath = ".\dist\$packageName.zip"
Compress-Archive -Path $packageDir -DestinationPath $zipPath -Force

Write-Host "`nPackage created successfully!" -ForegroundColor Green
Write-Host "Location: $zipPath" -ForegroundColor Green
Write-Host "Size: $([math]::Round((Get-Item $zipPath).Length / 1MB, 2)) MB" -ForegroundColor Green

Write-Host "`nYou can distribute the ZIP file or the folder:" -ForegroundColor Yellow
Write-Host "  $packageDir" -ForegroundColor Yellow
