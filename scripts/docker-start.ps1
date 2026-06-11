$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$npmRegistry = if ($env:NPM_REGISTRY) { $env:NPM_REGISTRY } else { "https://registry.npmmirror.com" }

Push-Location $repoRoot
try {
    $env:DOCKER_BUILDKIT = "0"

    docker build `
        -t rag-eval-backend:python3.12-slim `
        -f Dockerfile `
        .

    docker build `
        --build-arg "NPM_REGISTRY=$npmRegistry" `
        -t rag-eval-frontend:nginx-1.29.8 `
        -f web/Dockerfile `
        .

    docker compose up -d --no-build
    docker compose ps
}
finally {
    Pop-Location
}
