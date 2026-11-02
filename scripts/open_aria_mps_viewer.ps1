[CmdletBinding()]
param(
    [string]$VrsPath = '',
    [switch]$EyeGaze,
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $VrsPath) {
    $VrsPath = Join-Path $projectRoot 'data\Aria\raw_vrs\unknown_20260803_164557.vrs'
}
$vrsItem = Get-Item -LiteralPath $VrsPath
if ($vrsItem.PSIsContainer -or $vrsItem.Extension -ne '.vrs') {
    throw 'VrsPath must identify an existing .vrs file.'
}
$mpsDir = Join-Path $vrsItem.DirectoryName ('mps_' + $vrsItem.BaseName + '_vrs')
$trajectoryPath = Join-Path $mpsDir 'slam\closed_loop_trajectory.csv'
$pointsPath = Join-Path $mpsDir 'slam\semidense_points.csv.gz'
$sourcePaths = @($vrsItem.FullName, $trajectoryPath, $pointsPath)
$gazePath = ''
$previewName = 'mps_slam_rgb'
$logName = 'export'
$previewDescription = 'official viewer_mps defaults; closed-loop trajectory + filtered point cloud + rectified RGB; full recording'
if ($EyeGaze) {
    $gazePath = Join-Path $mpsDir 'eye_gaze\general_eye_gaze.csv'
    $gazeSummaryPath = Join-Path $mpsDir 'eye_gaze\summary.json'
    $sourcePaths += @($gazePath, $gazeSummaryPath)
    $previewName = 'mps_slam_rgb_eyegaze'
    $logName = 'export_eyegaze'
    $previewDescription += '; general MPS eye gaze (not personalized); official projection defaults'
}
$viewerExe = 'E:\Aria\tools\rerun-cli-0.33.0-x86_64-pc-windows-msvc.exe'
$exporter = '/opt/aria-vrs/bin/viewer_mps'
if (-not (Test-Path -LiteralPath $viewerExe -PathType Leaf)) {
    throw "Rerun executable not found: $viewerExe"
}
foreach ($sourcePath in $sourcePaths) {
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Required local input missing: $sourcePath. This launcher never requests MPS or uploads recordings."
    }
}

function Get-SourceFingerprints {
    foreach ($sourcePath in $sourcePaths) {
        $item = Get-Item -LiteralPath $sourcePath
        [ordered]@{
            path = $item.FullName
            bytes = $item.Length
            mtime_utc = $item.LastWriteTimeUtc.ToString('o')
        }
    }
}

function Convert-ToWslPath([string]$WindowsPath) {
    if ($WindowsPath -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Expected a local Windows drive path: $WindowsPath"
    }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
}

$sourcesBefore = @(Get-SourceFingerprints)
$sourceSignature = ConvertTo-Json -InputObject $sourcesBefore -Compress
$outputDir = Join-Path $projectRoot ('outputs\aria_mps_viewer\' + $vrsItem.BaseName)
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$rrdPath = Join-Path $outputDir ($previewName + '.rrd')
$receiptPath = Join-Path $outputDir ($previewName + '.complete.json')
$reuse = $false
if ((Test-Path -LiteralPath $rrdPath) -and (Test-Path -LiteralPath $receiptPath)) {
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    $reuse = ($receipt.source_signature -eq $sourceSignature) -and
        ($receipt.exporter -eq $exporter) -and
        ($receipt.output_bytes -eq (Get-Item -LiteralPath $rrdPath).Length)
}

if (-not $reuse) {
    if ((Test-Path -LiteralPath $rrdPath) -or (Test-Path -LiteralPath $receiptPath)) {
        throw "An incomplete or different export already exists. It has been preserved; choose a new output location before exporting again: $outputDir"
    }
    $wslVrs = Convert-ToWslPath $vrsItem.FullName
    $wslTrajectory = Convert-ToWslPath $trajectoryPath
    $wslPoints = Convert-ToWslPath $pointsPath
    $wslRrd = Convert-ToWslPath $rrdPath
    $stdoutPath = Join-Path $outputDir ($logName + '.stdout.log')
    $stderrPath = Join-Path $outputDir ($logName + '.stderr.log')
    $wslArgs = @(
        '-d', 'Ubuntu-24.04', '-u', 'root', '--', $exporter,
        '--vrs', ('"' + $wslVrs + '"'),
        '--trajectory', ('"' + $wslTrajectory + '"'),
        '--points', ('"' + $wslPoints + '"'),
        '--rrd_output_path', ('"' + $wslRrd + '"')
    )
    if ($EyeGaze) {
        $wslGaze = Convert-ToWslPath $gazePath
        $wslArgs += @('--eyegaze', ('"' + $wslGaze + '"'))
        Write-Host 'Eye gaze: official general model output; no personalized calibration is applied.'
    }
    Write-Host 'Exporting with the official viewer_mps (local files only; no MPS submission).'
    Write-Host "Input: $($vrsItem.FullName)"
    Write-Host "Output: $rrdPath"
    Write-Host "Progress log: $stderrPath"
    # Preserve the official defaults: rectified RGB, image downsampling 4,
    # JPEG quality 75, confidence-filtered point cloud capped at 500,000 points.
    # All RGB timestamps are replayed; no time crop or scientific preprocessing.
    $exportProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath -Wait -PassThru
    if ($exportProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $rrdPath)) {
        throw "Official export failed (exit $($exportProcess.ExitCode)). Inspect $stderrPath"
    }
    if ((Get-Item -LiteralPath $rrdPath).Length -eq 0) {
        throw 'Official exporter returned an empty recording; not marking it complete.'
    }
    $sourcesAfter = @(Get-SourceFingerprints)
    if ((ConvertTo-Json -InputObject $sourcesAfter -Compress) -ne $sourceSignature) {
        throw 'An input changed during export; not marking the output complete.'
    }
    # Generated run receipt only; never alter VRS, SLAM outputs, or old viewers.
    [ordered]@{
        source_signature = $sourceSignature
        sources = $sourcesBefore
        exporter = $exporter
        projectaria_tools = '2.3.0'
        rerun_sdk = '0.33.0'
        output_path = $rrdPath
        output_bytes = (Get-Item -LiteralPath $rrdPath).Length
        completed_utc = [DateTime]::UtcNow.ToString('o')
        visualization = $previewDescription
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
} else {
    Write-Host "Reusing completed local export: $rrdPath"
}

if (-not $NoOpen) {
    # Interactive viewer explicitly requested by the user, not a background helper.
    $viewerProcess = Start-Process -FilePath $viewerExe -ArgumentList @(
        '--bind', '127.0.0.1', '--new', '--memory-limit', '8GB',
        ('"' + $rrdPath + '"')
    ) -WindowStyle Normal -PassThru
    Write-Host "Rerun MPS viewer PID: $($viewerProcess.Id)"
}
Write-Host "Ready: $rrdPath"
