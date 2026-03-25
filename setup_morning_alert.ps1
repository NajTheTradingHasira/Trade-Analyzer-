# APEX Alert Engine — Morning Scheduled Task
# Run this script ONCE in PowerShell (as Administrator) to register the morning alert task.
#
# What it does:
#   - Creates a scheduled task "APEX-Trade-Alerts-Morning"
#   - Runs at 8:30 AM ET every weekday (Mon-Fri)
#   - Executes: python patches/alerts.py --slack-only --min-score 7
#   - Working directory: C:\Users\najee\trade-analyzer
#   - Reads SLACK_WEBHOOK_URL from system environment
#
# This runs alongside the existing 4:30 PM task (APEX-Trade-Alerts).
# Morning scan catches overnight gaps and pre-market signals.
#
# To remove: Unregister-ScheduledTask -TaskName "APEX-Trade-Alerts-Morning" -Confirm:$false

$TaskName = "APEX-Trade-Alerts-Morning"
$WorkDir = "C:\Users\najee\trade-analyzer"
$PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source

if (-not $PythonPath) {
    Write-Host "ERROR: python not found in PATH. Set PythonPath manually." -ForegroundColor Red
    exit 1
}

Write-Host "Setting up APEX Morning Alert scheduled task..." -ForegroundColor Cyan
Write-Host "  Python:    $PythonPath"
Write-Host "  WorkDir:   $WorkDir"
Write-Host "  Schedule:  8:30 AM ET, Mon-Fri"
Write-Host ""

# Check SLACK_WEBHOOK_URL
$existingWebhook = [System.Environment]::GetEnvironmentVariable("SLACK_WEBHOOK_URL", "User")
if (-not $existingWebhook) {
    Write-Host "  WARNING: SLACK_WEBHOOK_URL not set in user environment." -ForegroundColor Yellow
    Write-Host "  Run setup_scheduled_alert.ps1 first, or set it manually:" -ForegroundColor Yellow
    Write-Host '  [System.Environment]::SetEnvironmentVariable("SLACK_WEBHOOK_URL", "your_url", "User")' -ForegroundColor Gray
} else {
    Write-Host "  SLACK_WEBHOOK_URL already set in environment" -ForegroundColor Green
}

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "  Removed existing task: $TaskName" -ForegroundColor Yellow
}

# Trigger: 8:30 AM ET, Monday through Friday
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "8:30AM"

# Action: run python with the alert script
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "patches/alerts.py --slack-only --min-score 7" `
    -WorkingDirectory $WorkDir

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Register the task (runs as current user)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Description "APEX Trade Analyzer - Morning pre-market alert scan. Catches overnight gaps and pre-market signals." `
    -RunLevel Limited

Write-Host ""
Write-Host "Task '$TaskName' registered successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Active alert schedule:" -ForegroundColor Cyan
Write-Host "  8:30 AM  APEX-Trade-Alerts-Morning  (pre-market)" -ForegroundColor White
Write-Host "  4:30 PM  APEX-Trade-Alerts           (market close)" -ForegroundColor White
Write-Host ""
Write-Host "Verify with:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTask -TaskName 'APEX-Trade-Alerts*' | Format-Table TaskName, State, NextRunTime"
Write-Host ""
Write-Host "Test run now:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "Remove with:" -ForegroundColor Cyan
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
