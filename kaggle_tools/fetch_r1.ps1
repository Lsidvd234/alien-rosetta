# Download the R1 run (results + weights + epoch checkpoints) from Kaggle and check it.
# Usage (any PowerShell window):
#   powershell -ExecutionPolicy Bypass -File D:\Downloads\fllm-project\new-language\kaggle_tools\fetch_r1.ps1
#   ... add -Status to only see whether it has finished
param([switch]$Status, [switch]$Force, [string]$Name = "r1-hard-aug-50k")
$env:PYTHONUTF8 = "1"
Set-Location D:\Downloads\fllm-project\new-language
$py = "D:\Downloads\fllm-project\.venv-vocab\Scripts\python.exe"
$extra = @()
if ($Status) { $extra += "--status" }
if ($Force) { $extra += "--force" }
& $py kaggle_tools\fetch_run.py $Name @extra
