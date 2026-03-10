# Create New Google Cloud Project for YouTube API

Complete walkthrough (~15 minutes)

## Step 1: Go to Google Cloud Console

1. Open: https://console.cloud.google.com
2. Sign in with: **fkozina92@gmail.com**
3. Click the **Project Selector** (top left, shows current project)
4. Click **NEW PROJECT** button

## Step 2: Create New Project

1. **Project name:** Enter: `Joe YouTube`
2. **Organization:** Leave blank or select your org
3. Click **CREATE**
4. Wait 30 seconds for project to be created
5. Click **SELECT PROJECT** (or the notification banner)

## Step 3: Enable YouTube API

1. Search box at top: type `YouTube Data API`
2. Click: **YouTube Data API v3** (first result)
3. Click: **ENABLE** button (big blue button)
4. Wait for it to load (~10 seconds)

## Step 4: Create OAuth Credentials

1. Click: **CREATE CREDENTIALS** (button on right)
2. Choose: **Application type** → **Desktop application**
3. Click: **CREATE**
4. You'll see: "Desktop app credentials created"

## Step 5: Configure OAuth Consent Screen

1. In left menu: Click **OAuth consent screen**
2. Choose: **User Type: External** (or Internal if available)
3. Click: **CREATE**
4. Fill in the form:
   - **App name:** Joe
   - **User support email:** fkozina92@gmail.com
   - **Developer contact info:** fkozina92@gmail.com
   - Click **SAVE AND CONTINUE**

5. **Scopes step:**
   - Click **ADD OR REMOVE SCOPES**
   - Search: `youtube.upload`
   - Check: `https://www.googleapis.com/auth/youtube.upload`
   - Click **UPDATE**
   - Click **SAVE AND CONTINUE**

6. **Test users step:**
   - Add your email: fkozina92@gmail.com
   - Click **ADD**
   - Click **SAVE AND CONTINUE**

## Step 6: Download Client Secret

1. In left menu: Click **Credentials**
2. Under "Desktop application": Click **Download** (⬇️ icon)
3. A JSON file downloads: `client_secret_*.json`
4. Save it to: `C:\Users\fkozi\joe\config\client_secret.json`
   - **Important:** Must be exactly `client_secret.json`

## Step 7: Verify Downloaded File

After downloading, verify it's in the right place:

```bash
ls -la config/client_secret.json
```

You should see the file (~1KB).

## Step 8: Test Authentication

Now test if it works:

```bash
python3 test_youtube_oauth_oob.py
```

You should see:
- Browser opens
- Google login screen
- Authorization request
- Copy authorization code
- Paste in terminal
- Success!

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't find YouTube API | Make sure you searched in the right console, not "APIs library" |
| "Quota exceeded" during OAuth | Wait a few minutes, Google rate-limits new projects |
| Browser doesn't open | Copy URL from terminal and paste manually in browser |
| "Invalid client" error | Make sure you downloaded the right file (should say `client_secret_*.json`) |
| Can't paste in terminal | Right-click → Paste |

## Expected Success

When it works:

```
[STEP 3] Starting Out-of-Band authentication...
[OK] Imported YouTubeAuthOOB
[OK] Created auth instance

[STEP 4] Getting credentials...
[Open browser for authorization...]

[After you authorize]
[OK] Credentials obtained!
[OK] YouTube service ready!
[OK] Found 0 channel(s):
```

(0 channels because this is a brand new project - you haven't created YouTube channels yet)

## Next: Upload Video

Once authentication works, you can upload videos:

```bash
python3 run_full_pipeline_demo.py
```

This will:
1. Generate a video script
2. Create audio (TTS)
3. Create video (2.6 MB)
4. Upload to YouTube
5. Create a new channel (YouTube auto-creates one)

---

**Questions?** See `docs/YOUTUBE_OAUTH_FIX.md` for additional help.
