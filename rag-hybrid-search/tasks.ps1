# PowerShell equivalent of the Makefile for Windows users without GNU Make.
#
# Usage:  ./tasks.ps1 <target> [args...]
# Targets mirror the Makefile: help, demo, up, down, install, test, lint,
# format, typecheck, check, seed, replay-fixtures, clean, clean-all.

param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'

function Show-Help {
    Write-Host "rag-hybrid-search - tasks.ps1 targets:"
    Write-Host ""
    Write-Host "  help              Show this help"
    Write-Host "  demo              Run in offline demo mode (no API keys)"
    Write-Host "  demo-down         Stop demo containers"
    Write-Host "  up                Run in live mode (needs .env keys)"
    Write-Host "  down              Stop live containers"
    Write-Host "  install           Install all dependencies"
    Write-Host "  test              Run the test suite"
    Write-Host "  lint              Run ruff lint check"
    Write-Host "  format            Run ruff format"
    Write-Host "  typecheck         Run mypy --strict"
    Write-Host "  check             lint + typecheck + test"
    Write-Host "  seed              Seed index from data/raw/"
    Write-Host "  replay-fixtures   Record Claude responses (needs real keys)"
    Write-Host "  clean             Wipe persistent indexes"
    Write-Host "  clean-all         Wipe indexes + venv + caches"
    Write-Host ""
    Write-Host "Quickstart:"
    Write-Host "  ./tasks.ps1 demo    # offline, no API keys, ~2 min"
    Write-Host "  ./tasks.ps1 up      # live mode, requires .env with API keys"
}

function Invoke-Demo {
    docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build
}

function Invoke-DemoDown {
    docker compose -f docker-compose.yml -f docker-compose.demo.yml down
}

function Invoke-Up {
    docker compose up --build
}

function Invoke-Down {
    docker compose down
}

function Invoke-Install {
    uv pip install -e ".[dev]"
}

function Invoke-Test {
    uv run pytest
}

function Invoke-Lint {
    uv run ruff check .
}

function Invoke-Format {
    uv run ruff format .
}

function Invoke-Typecheck {
    uv run mypy --strict src/
}

function Invoke-Check {
    Invoke-Lint
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Invoke-Typecheck
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Invoke-Test
}

function Invoke-Seed {
    uv run python scripts/seed.py
}

function Invoke-ReplayFixtures {
    uv run python scripts/build_replay_fixtures.py
}

function Invoke-Clean {
    if (Test-Path data/chroma) { Remove-Item -Recurse -Force data/chroma }
    if (Test-Path data/bm25_index.pkl) { Remove-Item -Force data/bm25_index.pkl }
}

function Invoke-CleanAll {
    Invoke-Clean
    foreach ($p in '.venv', '.ruff_cache', '.mypy_cache', '.pytest_cache', 'htmlcov') {
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
    }
    if (Test-Path .coverage) { Remove-Item -Force .coverage }
}

switch ($Target.ToLower()) {
    'help'            { Show-Help }
    'demo'            { Invoke-Demo }
    'demo-down'       { Invoke-DemoDown }
    'up'              { Invoke-Up }
    'down'            { Invoke-Down }
    'install'         { Invoke-Install }
    'test'            { Invoke-Test }
    'lint'            { Invoke-Lint }
    'format'          { Invoke-Format }
    'typecheck'       { Invoke-Typecheck }
    'check'           { Invoke-Check }
    'seed'            { Invoke-Seed }
    'replay-fixtures' { Invoke-ReplayFixtures }
    'clean'           { Invoke-Clean }
    'clean-all'       { Invoke-CleanAll }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host ""
        Show-Help
        exit 1
    }
}
