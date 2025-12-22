$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sharedPath = Join-Path $repoRoot 'shaders\shared.wgsl'
$organTablePath = Join-Path $repoRoot 'config\ORGAN_TABLE.csv'
$outPath = Join-Path $repoRoot 'docs\sensor_power_table.csv'

# Parse shader parameter1 (d[3].w) for amino acids 0..19
$lines = Get-Content -LiteralPath $sharedPath

$aminoParam1ByCode = @{}

for ($i = 0; $i -lt $lines.Count; $i++) {
    $m = [regex]::Match($lines[$i], '^\s*//\s*(\d+)\s+([A-Z])\s+-\s+(.+?)\s*$')
    if (-not $m.Success) { continue }

    $idx = [int]$m.Groups[1].Value
    $code = $m.Groups[2].Value

    # Find the following array<vec4...> line
    $j = $i + 1
    $dataLine = $null
    while ($j -lt $lines.Count) {
        $s = $lines[$j].Trim()
        if ($s.Length -eq 0) { $j++; continue }
        if ($s.StartsWith('//')) { break }
        if ($s.Contains('array<vec4<f32>,6>(')) { $dataLine = $s; break }
        $j++
    }

    if (-not $dataLine) { continue }

    $vec4Matches = [regex]::Matches($dataLine, 'vec4<f32>\(([^)]*)\)')
    if ($vec4Matches.Count -lt 4) {
        throw "Unexpected AMINO_DATA format at idx=${idx}"
    }

    # 4th vec4, 4th component is parameter1
    $d3 = $vec4Matches[3].Groups[1].Value
    $parts = $d3.Split(',') | ForEach-Object { $_.Trim() }
    if ($parts.Count -ne 4) {
        throw "Unexpected vec4 component count at idx=${idx}: $d3"
    }

    $param1 = [double]$parts[3]

    if ($idx -ge 0 -and $idx -le 19) {
        $aminoParam1ByCode[$code] = $param1
    }
}

$sensorNameToTypeId = @{
    'Alpha Sensor' = 22
    'Beta Sensor' = 23
    'Agent Alpha Sensor' = 34
    'Agent Beta Sensor' = 35
    'Trail Energy Sensor (alpha)' = 37
    'Trail Energy Sensor (beta)' = 37
    'Alpha Magnitude Sensor' = 38
    'Alpha Magnitude Sensor (var)' = 39
    'Beta Magnitude Sensor' = 40
    'Beta Magnitude Sensor (var)' = 41
}

$promoters = @{
    'V_Valine' = 'V'
    'M_Methionine' = 'M'
    'H_Histidine' = 'H'
    'Q_Glutamine' = 'Q'
}

$organRows = Import-Csv -LiteralPath $organTablePath

$outRows = New-Object System.Collections.Generic.List[object]

foreach ($r in $organRows) {
    $modifierIndex = [int]$r.Modifier
    $modifierCode = [string]$r.Modifier_AA

    foreach ($kv in $promoters.GetEnumerator()) {
        $promoterCol = $kv.Key
        $promoterCode = $kv.Value

        $organName = ([string]$r.$promoterCol).Trim()
        if (-not $sensorNameToTypeId.ContainsKey($organName)) { continue }

        $organTypeId = [int]$sensorNameToTypeId[$organName]

        $promoterParam1 = if ($aminoParam1ByCode.ContainsKey($promoterCode)) { [double]$aminoParam1ByCode[$promoterCode] } else { $null }
        $modifierParam1 = if ($aminoParam1ByCode.ContainsKey($modifierCode)) { [double]$aminoParam1ByCode[$modifierCode] } else { $null }

        $usesParam1Gain = $organTypeId -in 22, 23, 38, 39, 40, 41

        $combined = $null
        $gainAbs = $null
        $polarity = $null

        if ($usesParam1Gain -and $null -ne $promoterParam1 -and $null -ne $modifierParam1) {
            $combined = $promoterParam1 + $modifierParam1
            $gainAbs = [math]::Abs($combined)
            $polarity = if ($combined -ge 0) { 1 } else { -1 }
        }

        $outRows.Add([pscustomobject]@{
            sensor_kind = $organName
            organ_type_id = $organTypeId
            promoter_code = $promoterCode
            promoter_param1 = $promoterParam1
            modifier_index = $modifierIndex
            modifier_code = $modifierCode
            modifier_param1 = $modifierParam1
            combined_param1 = $combined
            gain_abs = $gainAbs
            polarity = $polarity
            notes = if ($usesParam1Gain) { 'env_dye_sensor (gain=abs(p+m), sign=sign(p+m))' } else { 'non_param1_sensor (gain not derived from promoter/modifier param1 in shader)' }
        })
    }
}

# Add Energy Sensor (type 24) as a standalone entry (not from organ table)
$outRows.Add([pscustomobject]@{
    sensor_kind = 'Energy Sensor'
    organ_type_id = 24
    promoter_code = ''
    promoter_param1 = ''
    modifier_index = ''
    modifier_code = ''
    modifier_param1 = ''
    combined_param1 = ''
    gain_abs = ''
    polarity = ''
    notes = 'energy_sensor (uses energy->signal mapping; not promoter/modifier param1 gain)'
})

$outRows = $outRows | Sort-Object sensor_kind, promoter_code, modifier_index

$outDir = Split-Path -Parent $outPath
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$outRows | Export-Csv -LiteralPath $outPath -NoTypeInformation
Write-Host "Wrote $($outRows.Count) rows to $outPath"
