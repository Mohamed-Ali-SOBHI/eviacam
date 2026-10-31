param([Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$entries = @(
    @('1 - Demonstration sans camera', 'demo', $false),
    @('2 - Essai IA 2 minutes', 'yunet', $false),
    @('3 - Comparaison Haar 2 minutes', 'haar', $false),
    @('4 - Lancer IA barre des taches', 'yunet', $true)
)
foreach ($entry in $entries) {
    $vbs = @'
Option Explicit
If WScript.Arguments.Named.Exists("validate") Then WScript.Quit 0
Dim fs, folder, args, shell, app, powershell
Set fs = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
folder = fs.GetParentFolderName(WScript.ScriptFullName)
powershell = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"
args = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & folder & "\launch-trial.ps1"" -Mode MODECONTINUOUS"
LAUNCH
'@
    $launch = if ($entry[1] -eq 'demo') {
        'shell.Run """" & powershell & """ " & args, 0, False'
    } else {
        # The installed eViacam also requests administrator rights. Ask Windows
        # normally; do not change UAC policy or attempt silent elevation.
        "Set app = CreateObject(`"Shell.Application`")`r`napp.ShellExecute powershell, args, folder, `"runas`", 0"
    }
    $continuousArg = if ($entry[2]) { ' -Continuous' } else { '' }
    $vbs = $vbs.Replace('MODE', $entry[1]).Replace('CONTINUOUS', $continuousArg).Replace('LAUNCH', $launch)
    [IO.File]::WriteAllText((Join-Path $Destination ($entry[0]+'.vbs')), $vbs, [Text.Encoding]::ASCII)
}
