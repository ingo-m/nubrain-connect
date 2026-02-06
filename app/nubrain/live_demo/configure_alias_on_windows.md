# Configure aliases on windwos

Source:
https://gemini.google.com/app/2064434a54d167a5

Step 1: Find and Edit Your PowerShell $PROFILE
First, you need to find and open your profile file. Open a PowerShell prompt (not cmd.exe). Check if you already have a profile file by running:

```
Test-Path $PROFILE
```

If it returns False, create the file with this command:
```
New-Item -Path $PROFILE -Type File -Force

notepad $PROFILE
```

Step 2: Add Your Commands to the $PROFILE

Paste the following code into the empty Notepad window. This is the PowerShell equivalent of your .bashrc setup.

```
# 1. Activate your virtual environment
# Note: We use the .ps1 (PowerShell) activation script
# C:\Users\nubra\Documents\eeg_venv\venv\Scripts\activate
C:\Users\nubra\Documents\eeg_venv\venv\Scripts\Activate.ps1

# 2. Create your alias
# In PowerShell, a function is safer and more powerful for aliases with arguments.
function nubrain {
    nubrain --live_demo --config C:\Users\nubra\OneDrive\EEG-WS-DSI-24\cache\cached_5_trials.pickle
}

function nubrain-1 {
    nubrain --live_demo --config C:\Users\nubra\OneDrive\EEG-WS-DSI-24\cache\cached_trial_001.pickle
}
function nubrain-2 {
    nubrain --live_demo --config C:\Users\nubra\OneDrive\EEG-WS-DSI-24\cache\cached_trial_002.pickle
}
function nubrain-3 {
    nubrain --live_demo --config C:\Users\nubra\OneDrive\EEG-WS-DSI-24\cache\cached_trial_003.pickle
}
function nubrain-4 {
    nubrain --live_demo --config C:\Users\nubra\OneDrive\EEG-WS-DSI-24\cache\cached_trial_004.pickle
}
function nubrain-5 {
    nubrain --live_demo --config C:\Users\nubra\OneDrive\EEG-WS-DSI-24\cache\cached_trial_005.pickle
}

# Optional: Add a message so you know it worked
Write-Host "✅ PowerShell Profile loaded. Venv active." -ForegroundColor Green
```

Step 3: CRITICAL "Gotcha" — Set the Execution Policy

The first time you restart PowerShell, it will probably fail with a red error message about "running scripts is disabled on this system."

By default, PowerShell blocks all scripts from running as a security measure. You need to change this setting one time.

    Close your current PowerShell window.

    Go to the Start Menu, type "PowerShell", right-click "Windows PowerShell", and select "Run as Administrator".

    In the Administrator shell, run this command to allow locally-created scripts (like your $PROFILE) to run:
    PowerShell

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

    It will ask you to confirm. Type Y and press Enter.

That's it! You can close the Administrator shell.
