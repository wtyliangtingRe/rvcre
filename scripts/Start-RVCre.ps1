param(
    [string]$WorkspacePath = "data",
    [string]$ConfigPath = "config/backends.json",
    [string]$ListenAddress = "127.0.0.1",
    [ValidateRange(1, 65535)][int]$Port = 7860,
    [switch]$NoBrowser,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv/Scripts/python.exe"

if (-not (Test-Path $PythonPath)) {
    Write-Host "RVCre 的本地 Python 环境尚未准备。请在项目根目录运行：" -ForegroundColor Yellow
    Write-Host "  powershell -NoProfile -File .\scripts\Setup-RVCre.ps1"
    exit 1
}

Push-Location $ProjectRoot
try {
    & $PythonPath -c "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(x) is not None for x in ['rvcre','gradio']) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "RVCre 或界面依赖尚未安装，请重新运行 .\scripts\Setup-RVCre.ps1。" -ForegroundColor Yellow
        exit 1
    }
    $LaunchArguments = @("-m", "rvcre", "ui", "--workspace", $WorkspacePath, "--config", $ConfigPath,
                         "--host", $ListenAddress, "--port", $Port)
    if ($NoBrowser) { $LaunchArguments += "--no-browser" }
    & $PythonPath @LaunchArguments @ExtraArguments
    $ResultCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $ResultCode
