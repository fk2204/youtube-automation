# YouTube Automation - Auto-Start Setup

## Option 1: Windows Task Scheduler (Recommended)

### Step-by-Step:

1. **Press `Win + R`**, type `taskschd.msc`, press Enter

2. **Click "Create Basic Task"** (right panel)

3. **Name:** `YouTubeAutomation`
   **Description:** `Starts YouTube video automation scheduler`

4. **Trigger:** Select "When I log on"

5. **Action:** Select "Start a program"

6. **Program/script:** `wscript.exe`
   **Arguments:** `"C:\Users\fkozi\youtube-automation\start_scheduler_hidden.vbs"`

7. **Check** "Open Properties dialog" → Finish

8. In Properties:
   - **General tab:** Check "Run with highest privileges"
   - **Conditions tab:** Uncheck "Start only if on AC power"
   - **Settings tab:** Check "Allow task to be run on demand"

9. Click **OK**

---

## Option 2: Startup Folder (Simpler)

1. Press `Win + R`, type `shell:startup`, press Enter

2. Create a shortcut to: `C:\Users\fkozi\youtube-automation\start_scheduler.bat`

3. Right-click shortcut → Properties → Run: **Minimized**

---

## Option 3: Run Manually (Current)

Open terminal and run:
```
cd C:\Users\fkozi\youtube-automation
python run.py daily-all
```

---

## Files Created

| File | Purpose |
|------|---------|
| `start_scheduler.bat` | Main startup script |
| `start_scheduler_hidden.vbs` | Runs bat file without visible window |

---

## Verify It's Running

After reboot, check:
1. Task Manager → Details → Look for `python.exe`
2. Run `python run.py status` to see scheduler status
