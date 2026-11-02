[CmdletBinding()]
param(
    [string]$VrsPath = '',
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
$viewerExe = 'E:\Aria\tools\rerun-cli-0.33.0-x86_64-pc-windows-msvc.exe'
if (-not (Test-Path -LiteralPath $viewerExe -PathType Leaf)) {
    throw "Rerun executable not found: $viewerExe"
}
$outputDir = Join-Path $projectRoot ('outputs\aria_viewer\' + $vrsItem.BaseName)
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$rrdPath = Join-Path $outputDir 'all_streams.rrd'
$receiptPath = Join-Path $outputDir 'all_streams.complete.json'
$reuse = $false
if ((Test-Path -LiteralPath $rrdPath) -and (Test-Path -LiteralPath $receiptPath)) {
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    $reuse = ($receipt.source_path -eq $vrsItem.FullName) -and
        ($receipt.source_bytes -eq $vrsItem.Length) -and
        (([DateTimeOffset]$receipt.source_mtime_utc).UtcDateTime.Ticks -eq $vrsItem.LastWriteTimeUtc.Ticks) -and
        ($receipt.output_bytes -eq (Get-Item -LiteralPath $rrdPath).Length)
}
if (-not $reuse) {
    # Never overwrite an incomplete or different existing export.
    if ((Test-Path -LiteralPath $rrdPath) -or (Test-Path -LiteralPath $receiptPath)) {
        throw "An existing export cannot be safely reused. Preserve it and choose a new output name in this script: $outputDir"
    }
    function Convert-ToWslPath([string]$WindowsPath) {
        if ($WindowsPath -notmatch '^([A-Za-z]):\\(.*)$') {
            throw "Expected a local Windows drive path: $WindowsPath"
        }
        return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
    }
    $wslVrs = Convert-ToWslPath $vrsItem.FullName
    $wslRrd = Convert-ToWslPath $rrdPath
    $stdoutPath = Join-Path $outputDir 'export.stdout.log'
    $stderrPath = Join-Path $outputDir 'export.stderr.log'
    $wslArgs = @(
        '-d', 'Ubuntu-24.04', '--', '/opt/aria-vrs/bin/aria_rerun_viewer',
        '--vrs', ('"' + $wslVrs + '"'),
        '--rrd-output-path', ('"' + $wslRrd + '"')
    )
    Write-Host "Exporting all supported recorded streams: $($vrsItem.FullName)"
    Write-Host "Output: $rrdPath"
    Write-Host "Progress log: $stderrPath"
    # No --enabled-streams filter and no time crop: retain the full recording.
    # The official viewer's visualization compression/subsampling remains unchanged.
    $exportProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath -PassThru
    $exportProcess.WaitForExit()
    if ($exportProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $rrdPath)) {
        throw "Aria export failed (exit $($exportProcess.ExitCode)). Inspect $stderrPath"
    }
    $after = Get-Item -LiteralPath $vrsItem.FullName
    if ($after.Length -ne $vrsItem.Length -or $after.LastWriteTimeUtc -ne $vrsItem.LastWriteTimeUtc) {
        throw 'The input file changed during export; not marking this output complete.'
    }
    # This is a generated run receipt, not a source/configuration file.
    [ordered]@{
        source_path = $vrsItem.FullName
        source_bytes = $vrsItem.Length
        source_mtime_utc = $vrsItem.LastWriteTimeUtc.ToString('o')
        output_path = $rrdPath
        output_bytes = (Get-Item -LiteralPath $rrdPath).Length
        completed_utc = [DateTime]::UtcNow.ToString('o')
        exporter = '/opt/aria-vrs/bin/aria_rerun_viewer'
        streams = 'all supported recorded streams; official visualization defaults'
        interval = 'full recording'
    } | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
} else {
    Write-Host "Reusing verified completed export: $rrdPath"
}
if (-not $NoOpen) {
    # This is the interactive viewer the user asked to see, not a background helper.
    $viewerProcess = Start-Process -FilePath $viewerExe -ArgumentList @(
        '--bind', '127.0.0.1', '--new', '--memory-limit', '8GB',
        ('"' + $rrdPath + '"')
    ) -WindowStyle Normal -PassThru
    Write-Host "Rerun viewer PID: $($viewerProcess.Id)"
}
Write-Host "Ready: $rrdPath"
