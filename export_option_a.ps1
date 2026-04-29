param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $projectDir "..\..")
$destRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)

New-Item -ItemType Directory -Force -Path $destRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $destRoot "stubs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $destRoot "tools") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $destRoot "src") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $destRoot "src\python") | Out-Null

Copy-Item -Force (Join-Path $projectDir "CMakeLists.txt") (Join-Path $destRoot "CMakeLists.txt")
Copy-Item -Force (Join-Path $projectDir "README.md") (Join-Path $destRoot "README.md")
Copy-Item -Force (Join-Path $projectDir "stubs\python_runner_stub.cpp") (Join-Path $destRoot "stubs\python_runner_stub.cpp")
Copy-Item -Force (Join-Path $repoRoot "tools\train_colmap_headless.cpp") (Join-Path $destRoot "tools\train_colmap_headless.cpp")
Copy-Item -Force (Join-Path $repoRoot "vcpkg.json") (Join-Path $destRoot "vcpkg.json")
Copy-Item -Force (Join-Path $repoRoot "LICENSE") (Join-Path $destRoot "LICENSE") -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $repoRoot "src\python\runner.hpp") (Join-Path $destRoot "src\python\runner.hpp")

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Target
    )

    robocopy $Source $Target /E /XD .git build out .vs /XF CMakeCache.txt | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed for $Source -> $Target with exit code $LASTEXITCODE"
    }
}

Copy-Tree (Join-Path $repoRoot "cmake") (Join-Path $destRoot "cmake")
Copy-Tree (Join-Path $repoRoot "external") (Join-Path $destRoot "external")
Copy-Tree (Join-Path $repoRoot "src\core") (Join-Path $destRoot "src\core")
Copy-Tree (Join-Path $repoRoot "src\geometry") (Join-Path $destRoot "src\geometry")
Copy-Tree (Join-Path $repoRoot "src\io") (Join-Path $destRoot "src\io")
Copy-Tree (Join-Path $repoRoot "src\training") (Join-Path $destRoot "src\training")

Write-Host ""
Write-Host "Standalone training repo exported to:"
Write-Host "  $destRoot"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd $destRoot"
Write-Host "  git init"
Write-Host "  git add ."
Write-Host "  git commit -m 'Initial standalone COLMAP training build'"
