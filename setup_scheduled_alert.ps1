# APEX Alert Engine — Windows Task Scheduler Setup
# Run this script ONCE in PowerShell (as Administrator) to register the daily alert task.
#
# What it does:
#   - Creates a scheduled task "APEX-Trade-Alerts"
#   - Runs at 4:30 PM ET every weekday (Mon-Fri)
#   - Executes: python patches/alerts.py --slack-only --min-score 7
#   - Working directory: C:\Users\najee\trade-analyzer
#   - Reads SLACK_WEBHOOK_URL from system environment
#
# To remove: Unregister-ScheduledTask -TaskName "APEX-Trade-Alerts" -Confirm:$false

$TaskName = "APEX-Trade-Alerts"
$WorkDir = "C:\Users\najee\trade-analyzer"
$PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source

if (-not $PythonPath) {
    Write-Host "ERROR: python not found in PATH. Set PythonPath manually." -ForegroundColor Red
    exit 1
}

Write-Host "Setting up APEX Trade Alert scheduled task..." -ForegroundColor Cyan
Write-Host "  Python:    $PythonPath"
Write-Host "  WorkDir:   $WorkDir"
Write-Host "  Schedule:  4:30 PM ET, Mon-Fri"
Write-Host ""

# Set SLACK_WEBHOOK_URL as a persistent user environment variable if not already set
$existingWebhook = [System.Environment]::GetEnvironmentVariable("SLACK_WEBHOOK_URL", "User")
if (-not $existingWebhook) {
    $webhook = Read-Host "Enter your SLACK_WEBHOOK_URL (or press Enter to skip)"
    if ($webhook) {
        [System.Environment]::SetEnvironmentVariable("SLACK_WEBHOOK_URL", $webhook, "User")
        Write-Host "  SLACK_WEBHOOK_URL saved to user environment" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: SLACK_WEBHOOK_URL not set. Alerts will skip Slack delivery." -ForegroundColor Yellow
    }
} else {
    Write-Host "  SLACK_WEBHOOK_URL already set in environment" -ForegroundColor Green
}

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "  Removed existing task: $TaskName" -ForegroundColor Yellow
}

# Trigger: 4:30 PM ET, Monday through Friday
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "4:30PM"

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
    -Description "APEX Trade Analyzer - Daily alert scan at market close. Sends high-conviction signals to Slack." `
    -RunLevel Limited

Write-Host ""
Write-Host "Task '$TaskName' registered successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Verify with:" -ForegroundColor Cyan
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host ""
Write-Host "Test run now:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "Remove with:" -ForegroundColor Cyan
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
