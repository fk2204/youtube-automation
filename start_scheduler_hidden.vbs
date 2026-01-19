' YouTube Automation - Start Scheduler Hidden (no visible window)
' This VBScript runs the batch file without showing a command window

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "C:\Users\fkozi\youtube-automation\start_scheduler.bat" & chr(34), 0, False
Set WshShell = Nothing
