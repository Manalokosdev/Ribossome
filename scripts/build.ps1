# Helper script to remove BOM and build the project
& "$PSScriptRoot\remove_bom.ps1"
cargo build
