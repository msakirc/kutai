<#
  install_yasar_autostart.ps1  —  Layer 0 auto-start for Yaşar Usta.

  Registers a Task Scheduler task that keeps the hub alive across reboots and
  crashes. Per the validated design (docs/superpowers/specs/2026-07-17-
  yasar-usta-always-live-singleton-design.md §4.4/§7): Task Scheduler @ elevated
  logon in the USER session (NOT a Session-0 service, which would break the
  S13/S14 presence sensors). Never-duplicates is guaranteed by the hub's named
  mutex; this task is the "always-lives" relauncher.

  RUN ONCE, ELEVATED:
    powershell -ExecutionPolicy Bypass -File scripts\install_yasar_autostart.ps1

  AFTER running, for reboot recovery WITHOUT a manual login:
    - Enable Windows auto-logon: run `netplwiz`, uncheck "Users must enter a
      user name and password", enter the password once. (Boot -> auto-logon ->
      the at-logon trigger fires -> hub starts in your session.)
    - Remove any start_kutai.vbs shortcut from `shell:startup` so there is
      exactly one trigger (the mutex covers a stray one, but keep it clean).

  UNDO:  Unregister-ScheduledTask -TaskName YasarUsta -Confirm:$false
#>

$ErrorActionPreference = "Stop"

$root     = "C:\Users\sakir\Dropbox\Workspaces\kutay"
$python   = Join-Path $root ".venv\Scripts\python.exe"
$taskName = "YasarUsta"

if (-not (Test-Path $python)) { throw "venv python not found: $python" }

# Action: launch the hub hidden from the project dir.
$action = New-ScheduledTaskAction -Execute $python -Argument "kutai_wrapper.py" -WorkingDirectory $root

# Trigger: at logon of THIS user (with auto-logon => effectively at every boot).
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

# Settings: relaunch on crash (nonzero exit) every 1 min; never a 2nd instance;
# survive battery/idle; no run-time cap.
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable
$settings.ExecutionTimeLimit = "PT0S"   # unlimited

# Principal: elevated, in the interactive user session (presence sensors + GPU).
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

# ── Watchdog task: catches a HUNG-but-alive hub (restart-on-failure can't).
# Runs every 3 min; kills the hub if hub.alive is stale > threshold, so the
# main task's restart-on-failure relaunches it.
$watchName   = "YasarUstaWatchdog"
$alivePath   = Join-Path $root "logs\hub.alive"
$watchAction = New-ScheduledTaskAction -Execute $python `
    -Argument "-m yasar_usta.watchdog --alive `"$alivePath`"" -WorkingDirectory $root
$watchTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$watchTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 3) `
    -RepetitionDuration (New-TimeSpan -Days 3650)).Repetition
$watchSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName $watchName -Action $watchAction -Trigger $watchTrigger `
    -Settings $watchSettings -Principal $principal -Force | Out-Null

Write-Host "OK: '$taskName' (at-logon, elevated, restart-on-failure, no-duplicate) + '$watchName' (hung-hub watchdog, every 3 min) registered."
Write-Host "Test now:  Start-ScheduledTask -TaskName $taskName   (the mutex makes it a no-op if the hub is already running)"
Write-Host "NEXT: enable auto-logon via netplwiz for reboot-without-login recovery."
