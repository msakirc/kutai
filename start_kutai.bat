@echo off
REM == Launch Yasar Usta DETACHED via Task Scheduler ==========================
REM DO NOT run `python -m yasar_usta` directly here. That tethers the hub to
REM THIS console window; closing the window sends CTRL_CLOSE and hard-kills the
REM whole tree (hub + orchestrator + llama-server). That is exactly how the hub
REM died on 2026-07-23. The scheduled task is spawned by the Task Scheduler
REM service (svchost) with no console, so no window-close can ever kill it.
REM
REM The task is registered by scripts\install_yasar_autostart.ps1 (in the hub
REM repo, run elevated once). If the hub is already up, /Run is a safe no-op
REM (the task's IgnoreNew + the singleton mutex prevent a second instance).
REM
REM Clear any deliberate-stop marker first: /shutdown_hub leaves a hub.stopped
REM file that makes the hub refuse to start (even under the keep-alive trigger).
REM Running THIS launcher is the explicit "start" that un-stops it.
del "%LOCALAPPDATA%\YasarUsta\hub\hub.stopped" 2>nul
schtasks /Run /TN "YasarUsta"
if %ERRORLEVEL%==0 (
  echo Hub start requested via Task Scheduler ^(detached, background^).
  echo Status: Telegram, or  schtasks /Query /TN YasarUsta
) else (
  echo(
  echo ERROR: scheduled task "YasarUsta" not found or failed to start.
  echo Register it once, elevated:
  echo   powershell -ExecutionPolicy Bypass -File "C:\Users\sakir\Dropbox\Workspaces\yasar_usta\scripts\install_yasar_autostart.ps1"
)
