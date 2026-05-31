@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_submission_zip.ps1" %*
pause
