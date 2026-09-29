# push_to_github.ps1
# Simple automated script to push the recruitment project to GitHub

$gitExe = "C:\Users\Dell\AppData\Local\GitHubDesktop\app-3.6.6\resources\app\git\cmd\git.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "      PUSH TASK 2 NEURAL NETWORK TO GITHUB" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$repoUrl = Read-Host "Enter your GitHub repository URL (e.g. https://github.com/hk301007/AIML-Recruitment-2026-Harshita.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "[!] No URL entered. Aborting push." -ForegroundColor Yellow
    Exit
}

Write-Host "`n[*] Configuring remote origin..." -ForegroundColor Green
& $gitExe remote remove origin 2>$null
& $gitExe remote add origin $repoUrl

Write-Host "[*] Pushing 'main' branch to GitHub..." -ForegroundColor Green
& $gitExe push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[+] SUCCESS! Your recruitment repository is now live on GitHub!" -ForegroundColor Green
    Write-Host "    Repository URL: $repoUrl" -ForegroundColor Cyan
} else {
    Write-Host "`n[!] Push failed. Check your GitHub repository permissions or use GitHub Desktop." -ForegroundColor Red
}
