@echo off
rem 全自动动漫人物训练集提取器 - 一键启动 (uv 隔离环境, 不污染本地 Python)
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 uv, 请先安装: winget install astral-sh.uv
    pause
    exit /b 1
)
echo 首次运行会自动创建隔离虚拟环境并下载依赖(约1-2GB), 请耐心等待...
uv run extractor.py
pause
