# CUDA-Q Docker Container Launch Script for Windows
# This script runs the CUDA-Q container with GPU support and mounts your notebook folder

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CUDA-Q Docker Container Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Docker is running
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check for NVIDIA GPU support in Docker
Write-Host "`nChecking GPU support..." -ForegroundColor Yellow
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: GPU support may not be available in Docker." -ForegroundColor Yellow
    Write-Host "Make sure Docker Desktop has GPU support enabled in Settings > Resources > GPU" -ForegroundColor Yellow
}

# Set paths
$notebookPath = "D:\Work\Github\2026-NVIDIA\tutorial_notebook"
$containerWorkdir = "/workspace/tutorial_notebook"

Write-Host "`nStarting CUDA-Q container with Jupyter Lab..." -ForegroundColor Green
Write-Host "Your notebook folder will be mounted at: $containerWorkdir" -ForegroundColor Green
Write-Host "`nAccess Jupyter Lab at: http://localhost:8888" -ForegroundColor Cyan

# Run the container with:
# --gpus all: Enable GPU access
# -p 8888:8888: Expose Jupyter port
# -v: Mount your notebook folder
# --name: Name the container for easy reference
docker run --gpus all -it --rm `
    -p 8888:8888 `
    -v "${notebookPath}:/workspace/tutorial_notebook" `
    --name cudaq-labs `
    nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.13.0 `
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

Write-Host "`nContainer stopped." -ForegroundColor Yellow
