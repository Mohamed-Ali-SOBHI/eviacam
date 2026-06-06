#ifndef AppVersion
  #define AppVersion "2.1.4.1"
#endif
#ifndef PackageRoot
  #error PackageRoot must be supplied with /DPackageRoot=...
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

#define AppName "Enable Viacam"
#define AppExeName "eviacam.exe"
#define AppPublisher "Enable Viacam contributors"

[Setup]
AppId={{C287A0CA-FB45-4DBB-B8CF-4BA3FC46BEE3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/Mohamed-Ali-SOBHI/eviacam
AppSupportURL=https://github.com/Mohamed-Ali-SOBHI/eviacam/issues
AppUpdatesURL=https://github.com/Mohamed-Ali-SOBHI/eviacam/releases
DefaultDirName={localappdata}\Programs\Enable Viacam
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile={#PackageRoot}\LICENSE.txt
OutputDir={#OutputDir}
OutputBaseFilename=eviacam-{#AppVersion}-windows-x64-setup
SetupIconFile=installer.ico
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
UninstallDisplayIcon={app}\{#AppExeName}
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#PackageRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent unchecked
