param(
    [string]$DataDir = "E:\Sphinx Corpus",
    [ValidateSet("atlas", "ledger", "depth", "manifest", "atlas-ledger")]
    [string]$Phase = "atlas-ledger"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Config = Join-Path $Root "configs\corpus\sphinx_corpus_v1.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment is missing: $Python"
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

function Invoke-CorpusPhase {
    param([string]$Name)

    & $Python -m sphinx_corpus.cli --config $Config --data-dir $DataDir $Name
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
