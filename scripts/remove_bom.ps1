# Remove BOM from simulation.wgsl before build
$shaderPath = "shaders\simulation.wgsl"
$bytes = [System.IO.File]::ReadAllBytes($shaderPath)

if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    Write-Host "Removing BOM from simulation.wgsl..."
    $bytes = $bytes[3..($bytes.Length-1)]
    [System.IO.File]::WriteAllBytes($shaderPath, $bytes)
    Write-Host "BOM removed successfully"
} else {
    Write-Host "No BOM found in simulation.wgsl"
}
