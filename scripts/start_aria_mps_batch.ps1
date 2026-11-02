[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot 'data\Aria\mps_batch_2026-09-26_manifest.json'
$batchDir = Join-Path $projectRoot 'outputs\aria_mps_batch\2026-09-26_slam_eye_gaze'
$launchPath = Join-Path $batchDir 'launch.json'
$exitPath = Join-Path $batchDir 'client_exit.json'
$batchMutex = $null
$mutexHeld = $false

try {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $inputDir = (Get-Item -LiteralPath $manifest.input_windows).FullName
    if ($inputDir -ne 'E:\New_research\data\Aria\raw_vrs' -or
        $manifest.input_wsl -ne '/mnt/e/New_research/data/Aria/raw_vrs' -or
        ($manifest.features -join ',') -ne 'SLAM,EYE_GAZE') {
        throw 'Batch input or feature scope differs from the approved batch.'
    }
    # Official directory input scans recursively. Verify that its entire VRS
    # inventory is exactly the approved snapshot, including paths and sizes.
    $actualFiles = @(Get-ChildItem -LiteralPath $inputDir -Recurse -File -Filter '*.vrs')
    if ($actualFiles.Count -ne $manifest.recordings.Count -or $actualFiles.Count -ne 29) {
        throw 'VRS inventory changed. Review the batch scope before uploading.'
    }
    foreach ($entry in $manifest.recordings) {
        if ($entry.name -notmatch '^unknown_\d{8}_\d{6}\.vrs$') {
            throw 'Unexpected recording name in manifest.'
        }
        $expectedPath = Join-Path $inputDir $entry.name
        $matches = @($actualFiles | Where-Object FullName -eq $expectedPath)
        if ($matches.Count -ne 1) { throw "Missing or misplaced recording: $($entry.name)" }
        $item = $matches[0]
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $item.Length -ne $entry.bytes -or
            $item.LastWriteTimeUtc.ToString('o') -ne $entry.mtime_utc) {
            throw "Recording changed since approval: $($entry.name)"
        }
    }
    $ariaWsl = (Get-Command wsl.exe -ErrorAction Stop).Source
    foreach ($program in @('aria_mps', 'run_vrs_health_check')) {
        & $ariaWsl -d Ubuntu-24.04 -u root -- /usr/bin/test -x "/opt/aria-mps/bin/$program"
        if ($LASTEXITCODE -ne 0) { throw "Official executable missing: $program" }
    }
    Write-Host 'Approved batch: 29 recordings, 24.49 GiB, SLAM + EYE_GAZE only.'
    Write-Host "Input: $inputDir"
    Write-Host 'Existing local SLAM results are reused. No --force, no health-check bypass.'
    if ($CheckOnly) {
        # This checks PATH without loading authentication or processing VRS.
        & $ariaWsl -d Ubuntu-24.04 -u root -- /bin/bash -lc 'source /opt/aria-mps/bin/activate && command -v aria_mps run_vrs_health_check'
        if ($LASTEXITCODE -ne 0) { throw 'Activated environment check failed.' }
        Write-Host 'PASS: approved inventory and environment checked; no login or upload.'
        exit 0
    }
    if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected) {
        throw 'Use an interactive terminal. Authentication must not be recorded or redirected.'
    }
    $batchMutex = New-Object System.Threading.Mutex($false, 'Local\Aria_MPS_20260926_SLAM_EYE_GAZE')
    try { $mutexHeld = $batchMutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $mutexHeld = $true }
    if (-not $mutexHeld) { throw 'This batch launcher is already running. Use its existing terminal.' }
    if (Test-Path -LiteralPath $launchPath) {
        throw 'This batch was already launched. Inspect its existing requests before restarting: eye gaze does not reuse remote requests, so blind restart can duplicate processing.'
    }
    New-Item -ItemType Directory -Path $batchDir -Force | Out-Null
    # A launch receipt is not evidence of successful authentication or upload.
    [ordered]@{
        batch_id = $manifest.batch_id
        launched_utc = [DateTime]::UtcNow.ToString('o')
        launcher_pid = $PID
        input_directory = $inputDir
        manifest = $manifestPath
        features = @('SLAM', 'EYE_GAZE')
        recordings = 29
        status = 'official_client_starting_authentication_or_processing_not_yet_verified'
        note = 'No passwords, passphrases, tokens, or terminal transcript are stored by this launcher.'
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchPath -Encoding UTF8
    $Host.UI.RawUI.WindowTitle = 'Aria MPS Batch - 29 recordings - SLAM + Eye Gaze'
    Write-Host ''
    Write-Host 'This command uploads the approved VRS recordings to Meta MPS.' -ForegroundColor Yellow
    Write-Host 'If asked, enter your saved-token encryption passphrase directly in the official prompt.'
    Write-Host 'Do not paste the passphrase, login code or password into chat.'
    Write-Host 'Keep this terminal open for automatic result downloads. Do not start a second copy.'
    Write-Host 'Results: each VRS sibling mps_<recording>_vrs/slam and eye_gaze directories.'
    Write-Host ''
    # Activate the venv: aria_mps calls run_vrs_health_check via PATH.
    # Fixed, reviewed input directory; no interpolated shell fragments or secrets.
    & $ariaWsl -d Ubuntu-24.04 -u root -- /bin/bash -lc 'source /opt/aria-mps/bin/activate && exec aria_mps single -i /mnt/e/New_research/data/Aria/raw_vrs --features SLAM EYE_GAZE'
    $clientExit = $LASTEXITCODE
    [ordered]@{
        exited_utc = [DateTime]::UtcNow.ToString('o')
        client_exit_code = $clientExit
        note = 'Client exit alone does not prove that every cloud feature succeeded. Check official status and downloaded summaries.'
    } | ConvertTo-Json | Set-Content -LiteralPath $exitPath -Encoding UTF8
    Write-Host 'Official client exited. Check feature statuses and output files before claiming completion.'
    exit $clientExit
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    if ($mutexHeld -and $batchMutex) { $batchMutex.ReleaseMutex() }
    if ($batchMutex) { $batchMutex.Dispose() }
}
