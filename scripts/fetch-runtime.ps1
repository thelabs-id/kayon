# Stage the bundled llama.cpp sidecar from the pinned upstream release (RUN-1).
#
# The binaries are ~88 MB and gitignored, so they are fetched rather than committed. That makes this
# script the thing standing between "an installer built from this repo" and "an installer built from
# whatever happened to be on someone's disk" -- which is exactly what Kayon shipped before it existed
# (an unidentifiable build reporting `version: 1`).
#
# The checksum is not a formality: it is the same gate every model download goes through (DL-3).
# A mismatch fails the build rather than shipping an unverified binary to users.
#
# Usage:  pwsh scripts/fetch-runtime.ps1 [-Force]

[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pin  = Get-Content (Join-Path $repo 'src-tauri/runtime-pin.json') -Raw | ConvertFrom-Json
$dest = Join-Path $repo 'src-tauri/binaries/llama'
$stamp = Join-Path $dest '.llamacpp_version'

# Already staged at this exact pin? Nothing to do.
if (-not $Force -and (Test-Path $stamp)) {
    $have = (Get-Content $stamp -Raw).Trim()
    if ($have -eq $pin.sha256 -and (Test-Path (Join-Path $dest 'llama-server.exe'))) {
        Write-Host "runtime already staged: $($pin.tag) ($($pin.backend))"
        exit 0
    }
}

$url = "$($pin.upstream)/releases/download/$($pin.tag)/$($pin.asset)"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) $pin.asset
Write-Host "fetching $($pin.asset) from $($pin.tag)"
Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing

$actual = (Get-FileHash -Algorithm SHA256 -Path $tmp).Hash.ToLower()
if ($actual -ne $pin.sha256.ToLower()) {
    Remove-Item $tmp -Force
    throw "checksum mismatch for $($pin.asset)`n  expected $($pin.sha256)`n  got      $actual`nRefusing to stage an unverified runtime."
}
Write-Host "  sha256 ok"

$unzip = Join-Path ([System.IO.Path]::GetTempPath()) "kayon-rt-$($pin.tag)"
if (Test-Path $unzip) { Remove-Item $unzip -Recurse -Force }
Expand-Archive -Path $tmp -DestinationPath $unzip -Force

New-Item -ItemType Directory -Force -Path $dest | Out-Null
Get-ChildItem $dest -File | Remove-Item -Force
foreach ($f in $pin.files) {
    $src = Get-ChildItem -Path $unzip -Filter $f -Recurse -File | Select-Object -First 1
    if (-not $src) { throw "$f is missing from $($pin.asset) -- the pin's file list and the upstream zip disagree." }
    Copy-Item $src.FullName (Join-Path $dest $f) -Force
}

# The stamp records what is on disk, so a stale staging is detectable.
Set-Content -Path $stamp -Value $pin.sha256 -NoNewline -Encoding ascii
Remove-Item $tmp -Force
Remove-Item $unzip -Recurse -Force
Write-Host "staged $($pin.files.Count) files -> src-tauri/binaries/llama ($($pin.tag), $($pin.backend))"
