$ErrorActionPreference = "Stop"

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$shortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Abnormal Behavior Detection.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $projectPath "launch_app.bat"
$shortcut.WorkingDirectory = $projectPath
$shortcut.Description = "Launch the abnormal behavior detection web application"
$shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,220"
$shortcut.Save()

Write-Host "Desktop shortcut created: $shortcutPath"