param([string]$Python = "")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        if ($Python) {
            & $Python -m venv .venv
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.11 -m venv .venv
        } else {
            throw "请安装 Python 3.11，或用 -Python 指定 Python 3.11/3.12 的路径。"
        }
        if ($LASTEXITCODE -ne 0) { throw "无法创建 .venv；请检查 Python 3.11 是否已安装。" }
    }
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip 更新失败。" }
    & .\.venv\Scripts\python.exe -m pip install -e ".[ui]"
    if ($LASTEXITCODE -ne 0) { throw "应用依赖安装失败；请查看上方错误。" }
    Write-Host ""
    Write-Host "RVCre 界面与音频分析已安装。"
    Write-Host "启动：.\scripts\Start-RVCre.cmd"
    Write-Host "模型环境与权重按 README 分别准备，不会在这一步自动下载。"
} finally {
    Pop-Location
}
