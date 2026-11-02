param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".conda-envs\phase1-py310\Scripts\python.exe"
$Pipeline = Join-Path $PSScriptRoot "phase1_pipeline.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Phase-1 Python environment not found: $Python"
}

& $Python -u $Pipeline @PipelineArgs
exit $LASTEXITCODE
