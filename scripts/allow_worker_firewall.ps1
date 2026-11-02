# Run once from an elevated PowerShell. Inbound access for these executables only.
# Dota also listens on wildcard/IPv6 sockets, so loopback-only rules do not cover
# all listeners and can leave Windows asking for permission.
# Stable executable paths are reused by every future collection run.
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
for ($workerIndex = 0; $workerIndex -lt 20; $workerIndex++) {
    $workerName = '{0:D2}' -f $workerIndex
    $program = "C:\AMI\workers\$workerName\game\bin\win64\dota2.exe"
    $ruleName = "SF-Duel-Worker-$workerName-Local"
    if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
        Set-NetFirewallRule -Name $ruleName -DisplayName "SF Duel worker $workerName" `
            -Enabled True -Direction Inbound `
            -Action Allow -Profile Any -Program $program -Protocol Any `
            -LocalAddress Any -RemoteAddress Any
    } else {
        New-NetFirewallRule -Name $ruleName -DisplayName "SF Duel worker $workerName" `
            -Direction Inbound -Action Allow -Profile Any -Program $program `
            -RemoteAddress Any | Out-Null
    }
}
$rules = @(Get-NetFirewallRule -PolicyStore ActiveStore -Name 'SF-Duel-Worker-*-Local')
if ($rules.Count -ne 20 -or @($rules | Where-Object {
    $_.Enabled -ne 'True' -or $_.Action -ne 'Allow' -or $_.Direction -ne 'Inbound'
}).Count -ne 0) {
    throw 'The 20 inbound allow rules were not found in the active firewall policy.'
}
Write-Output 'Verified 20 active inbound allow rules for the stable SF Duel worker paths (all remote addresses, all profiles).'
