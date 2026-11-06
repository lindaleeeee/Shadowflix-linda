@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul && (python shadow_server.py) || (py shadow_server.py)
pause
