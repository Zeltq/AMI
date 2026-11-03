param([int]$TargetPid,[string]$OutputPath)
$ErrorActionPreference='Stop'
$writer=[System.IO.StreamWriter]::new($OutputPath,$false,[System.Text.UTF8Encoding]::new($false))
try {
    while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
        $rows=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {$_.Name -like "pid_${TargetPid}_*"})
        $adapters=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory)
        $entry=@{unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();pid=$TargetPid;dedicated_bytes=($rows|Measure-Object -Property DedicatedUsage -Sum).Sum;shared_bytes=($rows|Measure-Object -Property SharedUsage -Sum).Sum;committed_bytes=($rows|Measure-Object -Property TotalCommitted -Sum).Sum;adapter_dedicated_bytes=($adapters|Measure-Object -Property DedicatedUsage -Sum).Sum}
        $writer.WriteLine(($entry|ConvertTo-Json -Compress));$writer.Flush()
        Start-Sleep -Seconds 2
    }
} finally {$writer.Dispose()}
