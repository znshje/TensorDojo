@echo off
chcp 65001 >nul
setlocal EnableExtensions

rem 切换到脚本所在目录，保证相对路径可用
cd /d "%~dp0"

set "PYTHON_BIN="
set "PYTHON_ARGS="

if defined DOJO_PYTHON (
  set "PYTHON_BIN=%DOJO_PYTHON%"
  call :check_python
  if errorlevel 1 (
    echo [Tensor Dojo] DOJO_PYTHON 指向的 Python 未安装 PyTorch：%DOJO_PYTHON%
    exit /b 1
  )
  goto :python_ok
)

set "PYTHON_BIN=python"
call :check_python
if not errorlevel 1 goto :python_ok

set "PYTHON_BIN=py"
set "PYTHON_ARGS=-3"
call :check_python
if not errorlevel 1 goto :python_ok

echo [Tensor Dojo] 未找到已安装 PyTorch 的 Python 环境。
echo 请安装 Python 3.10+ 与 PyTorch 2.2+，或设置 DOJO_PYTHON 指向带 PyTorch 的 python.exe。
exit /b 1

:check_python
"%PYTHON_BIN%" %PYTHON_ARGS% -c "import torch" >nul 2>&1
exit /b %errorlevel%

:python_ok
echo [Tensor Dojo] 使用 Python：%PYTHON_BIN% %PYTHON_ARGS%

"%PYTHON_BIN%" %PYTHON_ARGS% -c "import jedi" >nul 2>&1
if errorlevel 1 (
  echo [Tensor Dojo] 缺少代码补全依赖 jedi，正在安装…
  "%PYTHON_BIN%" %PYTHON_ARGS% -m pip install "jedi>=0.19"
  if errorlevel 1 (
    echo [Tensor Dojo] jedi 安装失败，代码补全将使用内置提示。
  )
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [Tensor Dojo] 未找到 npm，请安装 Node.js 20.19+ / 22.12+ 并加入 PATH。
  exit /b 1
)

if not exist "node_modules" (
  echo [Tensor Dojo] 首次启动，正在安装 npm 依赖…
  call npm ci --no-audit --no-fund
  if errorlevel 1 (
    echo [Tensor Dojo] npm ci 失败，请检查网络或 Node.js 版本。
    exit /b 1
  )
)

echo [Tensor Dojo] 正在构建前端…
call npm run build
if errorlevel 1 (
  echo [Tensor Dojo] 前端构建失败。
  exit /b 1
)

if not defined DOJO_PORT set "DOJO_PORT=8765"
echo [Tensor Dojo] 启动服务：http://127.0.0.1:%DOJO_PORT%
"%PYTHON_BIN%" %PYTHON_ARGS% backend/server.py --port %DOJO_PORT%
