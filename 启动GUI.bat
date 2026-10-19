@echo off
rem 切换 UTF-8 代码页, 修复中文显示乱码
chcp 65001 >nul
cd /d "%~dp0"

rem ===== 所有缓存/模型/虚拟环境都放在本 repo 目录内, 不占用 C 盘 =====
set "UV_CACHE_DIR=%~dp0runtime\uv_cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0runtime\python"
set "HF_HOME=%~dp0runtime\hf_cache"
set "HF_HUB_CACHE=%~dp0runtime\hf_cache\hub"
set "XDG_CACHE_HOME=%~dp0runtime\xdg_cache"
set "GRADIO_TEMP_DIR=%~dp0runtime\gradio_tmp"
rem 如需 HuggingFace 镜像加速, 取消下一行注释:
rem set "HF_ENDPOINT=https://hf-mirror.com"

where uv >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 uv, 请先安装: winget install astral-sh.uv
    pause
    exit /b 1
)

echo 首次运行会在本文件夹内创建隔离环境(runtime\ 和 .venv\)并下载依赖与模型, 全部存放在当前目录, 不会占用 C 盘, 请耐心等待...
uv run extractor.py
pause
