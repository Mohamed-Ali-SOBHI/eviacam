param(
    [ValidateSet('yunet','haar','demo')][string]$Mode = 'yunet',
    [switch]$Continuous
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$folder = $PSScriptRoot
$original = Join-Path ${env:ProgramFiles(x86)} 'Enable Viacam\bin\eviacam.exe'
$trialExe = Join-Path $folder 'eviacam.exe'
Set-Location -LiteralPath $folder
$guard = New-Object System.Threading.Mutex($false, 'Local\EviacamYuNetTrialSupervisor')
if (-not $guard.WaitOne(0)) {
    [System.Windows.Forms.MessageBox]::Show('Un essai ou une demonstration est deja en cours.', 'eViacam') | Out-Null
    exit
}
$trial = $null
$restore = $false
$failure = $null
try {
    if ($Mode -ne 'demo') {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        try {
            $principal = [Security.Principal.WindowsPrincipal]::new($identity)
            if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
                throw 'Utilisez le lanceur VBS et acceptez la confirmation Windows. Le controle de certaines interfaces Windows demande les memes droits que votre eViacam original.'
            }
        } finally { $identity.Dispose() }
    }
    $env:EVIACAM_FACE_DETECTOR = if ($Mode -eq 'haar') { 'haar' } else { 'yunet' }
    $env:EVIACAM_TIMED_TRIAL = '1'
    if ($Mode -eq 'demo') {
        $env:EVIACAM_REPLAY = 'replay.avi'
        $trial = Start-Process -FilePath $trialExe -ArgumentList '-c', ('"'+(Join-Path $folder 'demo.ini')+'"') -WorkingDirectory $folder -WindowStyle Normal -PassThru
        $trial.WaitForExit()
        exit
    }
    if (-not (Test-Path -LiteralPath $original)) { throw 'Installation originale eViacam introuvable.' }
    $duration = if ($Continuous) { 'Session sans limite de temps' } else { 'Essai de 2 minutes' }
    $returnMessage = if ($Continuous) { 'Votre version habituelle sera relancee quand vous fermerez YuNet.' } else { "Votre version habituelle sera relancee apres 2 minutes, ou des que vous fermerez l'essai." }
    $message = "$duration : $Mode.`n`n1. Cliquez sur OK.`n2. Fermez votre eViacam habituel et confirmez sa fermeture.`n3. La version d'essai demarrera automatiquement, avec vos reglages copies.`n`n$returnMessage Vous avez 5 minutes pour fermer l'ancienne version."
    $answer = [System.Windows.Forms.MessageBox]::Show($message, 'eViacam - essai avec retour automatique', 'OKCancel', 'Information')
    if ($answer -ne 'OK') { exit }
    $deadline = [DateTime]::UtcNow.AddMinutes(5)
    while (Get-Process -Name eviacam -ErrorAction SilentlyContinue) {
        if ([DateTime]::UtcNow -gt $deadline) { exit }
        Start-Sleep -Milliseconds 500
    }
    $restore = $true
    Start-Sleep -Seconds 2
    Remove-Item Env:EVIACAM_REPLAY -ErrorAction SilentlyContinue
    $trial = Start-Process -FilePath $trialExe -WorkingDirectory $folder -WindowStyle Normal -PassThru
    if ($Continuous) {
        $trial.WaitForExit()
    } elseif (-not $trial.WaitForExit(120000)) {
        $null = $trial.CloseMainWindow()
        if (-not $trial.WaitForExit(5000)) {
            # Only the process created by this supervisor, never the installed app.
            Stop-Process -Id $trial.Id -ErrorAction SilentlyContinue
            $trial.WaitForExit(5000) | Out-Null
        }
    }
} catch {
    $failure = $_.Exception.Message
} finally {
    Remove-Item Env:EVIACAM_FACE_DETECTOR,Env:EVIACAM_TIMED_TRIAL,Env:EVIACAM_REPLAY -ErrorAction SilentlyContinue
    if ($restore -and -not (Get-Process -Name eviacam -ErrorAction SilentlyContinue)) {
        Start-Sleep -Seconds 2
        Start-Process -FilePath $original -WorkingDirectory (Split-Path $original) -WindowStyle Normal
    }
    $guard.ReleaseMutex()
    $guard.Dispose()
}
if ($failure) {
    [System.Windows.Forms.MessageBox]::Show($failure, 'eViacam - essai') | Out-Null
}
