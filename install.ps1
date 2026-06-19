# install.ps1 — Install MAGMA Memory Provider for Hermes Agent (Windows)
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1
# Or just right-click and "Run with PowerShell"

param(
    [string]$HermesHome = "$env:USERPROFILE\.hermes"
)

$PluginDir = "$HermesHome\plugins\magma"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDir = Join-Path $ScriptDir "magma"

Write-Host "🔧 Installing MAGMA Memory Provider..." -ForegroundColor Cyan
Write-Host "   Target: $PluginDir"

# Ensure plugins directory exists
New-Item -ItemType Directory -Force -Path "$HermesHome\plugins" | Out-Null

# Check if already installed
if (Test-Path $PluginDir) {
    $choice = Read-Host "   ⚠  Plugin already exists. Overwrite? [y/N]"
    if ($choice -ne "y" -and $choice -ne "Y") {
        Write-Host "   ✗ Cancelled" -ForegroundColor Red
        exit 1
    }
    Remove-Item -Recurse -Force $PluginDir
}

# Copy plugin files
Copy-Item -Recurse -Path $SourceDir -Destination $PluginDir
Write-Host "   ✓ Files copied" -ForegroundColor Green

# Install Python dependencies
Write-Host "   📦 Installing Python dependencies..." -ForegroundColor Yellow
try {
    pip install numpy networkx 2>&1 | Out-Null
    Write-Host "   ✓ Dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "   ⚠  pip install failed. Run manually: pip install numpy networkx" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the provider:"
Write-Host "     hermes config set memory.provider magma"
Write-Host ""
Write-Host "  2. Restart Hermes (or /reset in an active session)"
Write-Host ""
Write-Host "  3. (Optional) For better semantic embeddings:"
Write-Host "     pip install sentence-transformers"
Write-Host ""
Write-Host "See README.md for full documentation."
