# Fix YouTube OAuth "redirect_uri_mismatch" Error

## Problem
Error: `Error 400: redirect_uri_mismatch`

This happens when the OAuth app in Google Cloud Console doesn't have the correct redirect URI configured.

## Solution (5 minutes)

### Step 1: Open Google Cloud Console
1. Go to https://console.cloud.google.com
2. Make sure you're logged in with **fkozina92@gmail.com**
3. Select project: **fresh-geography-486523-f1**

### Step 2: Configure OAuth Consent Screen
1. Click **APIs & Services** in the left menu
2. Click **OAuth consent screen**
3. Make sure:
   - **User Type**: Internal (if available) or External
   - **App name**: Joe (or whatever you want)
   - **User support email**: fkozina92@gmail.com
   - **Developer contact**: fkozina92@gmail.com
4. Click **Save and Continue**

### Step 3: Add Authorized Redirect URI
1. Click **Credentials** in the left menu
2. You should see your OAuth app (Client ID: 34195007913-...)
3. Click on it to edit
4. Scroll down to **Authorized redirect URIs**
5. Add these URIs:
   ```
   http://localhost:8080/
   http://localhost:8080
   http://127.0.0.1:8080/
   http://127.0.0.1:8080
   urn:ietf:wg:oauth:2.0:oob
   ```
6. Click **Save**

### Step 4: Verify OAuth Type
Make sure your credentials are **OAuth 2.0 Desktop Application** (not Web):
1. Click **Credentials**
2. Find your OAuth app
3. Look at **Application type** - should say "Desktop app"
4. If it says "Web application", you may need to create a new Desktop credential

### Step 5: Test Again
Run the test:
```bash
python3 test_youtube_oauth.py
```

## If Still Getting Error

### Option A: Use a Different Port
If port 8080 is blocked or used by another app:
1. Port 8080 is tried first
2. If blocked, it will auto-select another port
3. But that won't work unless you also add dynamic URIs

### Option B: Re-download client_secret.json
Sometimes the cached version is stale:
1. Go to Google Cloud Console → Credentials
2. Find your OAuth app
3. Click the download icon (⬇️)
4. Save as `config/client_secret.json`
5. Try again

### Option C: Create New OAuth Credentials
If nothing works, create fresh credentials:
1. Go to Google Cloud Console → Credentials
2. Delete the old OAuth app
3. Click **Create Credentials** → **OAuth 2.0 Desktop Application**
4. Name: "Joe YouTube Uploader"
5. Add redirect URIs:
   - http://localhost:8080/
   - http://127.0.0.1:8080/
   - urn:ietf:wg:oauth:2.0:oob
6. Download and save as `config/client_secret.json`
7. Try again

## Expected Success Output
When it works, you'll see:
```
[OK] Authentication successful!
[OK] YouTube service ready!
[OK] Found 3 channel(s):
     - money_blueprints
     - mind_unlocked
     - untold_stories
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| Address already in use | Port 8080 is busy, close other apps using it |
| Client secrets file not found | Make sure config/client_secret.json exists |
| Invalid client | client_secret.json is corrupted, re-download |
| redirect_uri_mismatch | Add http://localhost:8080/ to Google Cloud redirect URIs |

## Still Not Working?

Email Google Support or check these resources:
- https://developers.google.com/youtube/v3/guides/authentication
- https://cloud.google.com/docs/authentication/oauth2#desktop

