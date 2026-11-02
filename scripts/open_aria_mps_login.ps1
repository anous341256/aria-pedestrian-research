[CmdletBinding()]
param(
    [switch]$CheckOnly
)

# Deliberately separate authentication from MPS processing. Never accept a VRS
# path, invoke a processing request, enable debug logging, or save credentials.
$ErrorActionPreference = 'Stop'
$ariaDistro = 'Ubuntu-24.04'
$ariaEnv = '/opt/aria-mps'
$ariaAccount = $null

try {
    $ariaWsl = (Get-Command wsl.exe -ErrorAction Stop).Source
    foreach ($ariaProgram in @('python', 'aria_mps', 'projectaria_login', 'run_vrs_health_check')) {
        & $ariaWsl -d $ariaDistro -- /usr/bin/test -x "$ariaEnv/bin/$ariaProgram"
        if ($LASTEXITCODE -ne 0) {
            throw "Missing executable: $ariaEnv/bin/$ariaProgram in $ariaDistro."
        }
    }

    if ($CheckOnly) {
        Write-Host 'Offline checks only: no login, no credentials read, no VRS processing.'
        & $ariaWsl -d $ariaDistro -- "$ariaEnv/bin/python" -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
        & $ariaWsl -d $ariaDistro -- "$ariaEnv/bin/aria_mps" --help
        if ($LASTEXITCODE -ne 0) { throw 'MPS CLI help failed.' }
        & $ariaWsl -d $ariaDistro -- "$ariaEnv/bin/aria_mps" single --help
        if ($LASTEXITCODE -ne 0) { throw 'Single-recording CLI help failed.' }
        & $ariaWsl -d $ariaDistro -- "$ariaEnv/bin/projectaria_login" --help
        if ($LASTEXITCODE -ne 0) { throw 'Login CLI help failed.' }
        & $ariaWsl -d $ariaDistro -- "$ariaEnv/bin/run_vrs_health_check" --list-configurations
        if ($LASTEXITCODE -ne 0) { throw 'Native health-check component failed.' }
        Write-Host 'PASS: MPS installation checks completed. Account and cloud processing are untested.'
        exit 0
    }

    if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected) {
        throw 'Login requires an interactive, non-recorded terminal. Double-click Open_Aria_MPS_Login.cmd.'
    }

    Write-Host 'Project Aria MPS - official authentication only'
    Write-Host 'Environment: Ubuntu-24.04 /opt/aria-mps (WSL storage on drive E:)'
    Write-Host 'No VRS file will be opened or uploaded. No transcript is created.'
    Write-Host ''
    Write-Host 'Enter the Aria account associated with your glasses.'
    Write-Host 'For a managed Gen 2 account, the official tool will display a URL and code.'
    Write-Host 'Open https://work.meta.com/cli in your Windows browser and enter that code yourself.'
    Write-Host 'For a legacy account, enter its password only at the official hidden prompt.'
    Write-Host 'If asked for an encryption passphrase, choose and remember it yourself (12+ characters).'
    Write-Host 'The official tool may store a login token in the WSL keyring or an encrypted file.'
    Write-Host 'Do not paste passwords, authorization codes, or passphrases into chat.'
    Write-Host ''

    $ariaAccount = (Read-Host 'Aria username/email (blank cancels)').Trim()
    if ([string]::IsNullOrWhiteSpace($ariaAccount)) {
        Write-Host 'Cancelled before authentication.'
        exit 0
    }
    # Keep the native Windows -> WSL argument boundary unambiguous. These are
    # account identifiers, not shell fragments; do not use bash -c or eval.
    if ($ariaAccount.Length -gt 254 -or $ariaAccount -notmatch '^[A-Za-z0-9][A-Za-z0-9._+@-]*$') {
        throw 'Unsupported account format. Use the Aria username or full email; do not include spaces or quotes.'
    }

    & $ariaWsl -d $ariaDistro -- "$ariaEnv/bin/projectaria_login" login --username $ariaAccount
    $ariaLoginExit = $LASTEXITCODE
    $ariaAccount = $null
    if ($ariaLoginExit -ne 0) {
        throw "Official login command exited with code $ariaLoginExit. No processing request was started."
    }

    Write-Host ''
    Write-Host 'Authentication command completed. Check the official output above for token persistence.'
    Write-Host 'No MPS processing was requested. Select and approve a recording separately before uploading.'
    exit 0
} catch {
    $ariaAccount = $null
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    $ariaAccount = $null
}
