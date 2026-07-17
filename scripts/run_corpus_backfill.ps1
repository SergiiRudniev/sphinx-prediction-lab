param(
    [string]$DataDir = "E:\Sphinx Corpus",
    [ValidateSet("atlas", "ledger", "depth", "manifest", "atlas-ledger")]
    [string]$Phase = "atlas-ledger",
    [ValidateSet("fast", "full")]
    [string]$Profile = "fast",
    [ValidateRange(0, 256)]
    [int]$Workers = 0,
    [ValidateRange(0, 100)]
    [double]$RequestsPerSecond = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ConfigName = if ($Profile -eq "fast") {
    "sphinx_corpus_s0_fast_v1.json"
} else {
    "sphinx_corpus_v1.json"
}
$Config = Join-Path $Root "configs\corpus\$ConfigName"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment is missing: $Python"
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

function Invoke-CorpusPhase {
    param([string]$Name)

    $Arguments = @(
        "-m", "sphinx_corpus.cli",
        "--config", $Config,
        "--data-dir", $DataDir,
        $Name
    )
    if ($Name -eq "ledger") {
        if ($Workers -gt 0) {
            $Arguments += @("--workers", "$Workers")
        }
        if ($RequestsPerSecond -gt 0) {
            $Arguments += @("--requests-per-second", "$RequestsPerSecond")
        }
    }

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Sphinx Corpus phase failed: $Name"
    }
}

if ($Phase -eq "atlas-ledger") {
    Invoke-CorpusPhase "atlas"
    Invoke-CorpusPhase "ledger"
    Invoke-CorpusPhase "manifest"
} else {
    Invoke-CorpusPhase $Phase
}
