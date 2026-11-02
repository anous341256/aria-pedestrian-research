# Interactive local secret entry. No key is written to console or command arguments.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$secretDirectory = Join-Path $env:LOCALAPPDATA 'CodexResearch'
$secretPath = Join-Path $secretDirectory 'roboflow_api_key.dpapi'
$statusPath = 'E:\New_research\outputs\sam3_vs_sam2_20261001\credential_status.json'

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Roboflow - Local API Key Setup'
$form.ClientSize = New-Object System.Drawing.Size(550,220)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.TopMost = $true
$form.ShowInTaskbar = $true

$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(20,20)
$label.Size = New-Object System.Drawing.Size(510,48)
$label.Text = "Paste your Roboflow Private API Key below.`r`nIt will be encrypted for your Windows account and used for local SAM3 setup."
$form.Controls.Add($label)

$inputBox = New-Object System.Windows.Forms.TextBox
$inputBox.Location = New-Object System.Drawing.Point(20,80)
$inputBox.Size = New-Object System.Drawing.Size(510,28)
$inputBox.UseSystemPasswordChar = $true
$form.Controls.Add($inputBox)

$status = New-Object System.Windows.Forms.Label
$status.Location = New-Object System.Drawing.Point(20,120)
$status.Size = New-Object System.Drawing.Size(510,35)
$form.Controls.Add($status)

$save = New-Object System.Windows.Forms.Button
$save.Text = 'Save encrypted key'
$save.Location = New-Object System.Drawing.Point(330,168)
$save.Size = New-Object System.Drawing.Size(200,32)
$save.Add_Click({
    if ([string]::IsNullOrWhiteSpace($inputBox.Text)) {
        $status.Text = 'Please paste your Private API Key.'
        return
    }
    try {
        [System.IO.Directory]::CreateDirectory($secretDirectory) | Out-Null
        $secureValue = ConvertTo-SecureString -String $inputBox.Text.Trim() -AsPlainText -Force
        $encryptedValue = ConvertFrom-SecureString -SecureString $secureValue
        [System.IO.File]::WriteAllText($secretPath, $encryptedValue)
        $inputBox.Clear()
        $secureValue.Dispose()
        $encryptedValue = $null
        [System.IO.File]::WriteAllText($statusPath, '{"status":"saved_locally_encrypted"}')
        $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $form.Close()
    } catch {
        $status.Text = 'Could not save the key. Close this window and report this message.'
    }
})
$form.Controls.Add($save)
$form.AcceptButton = $save
$form.Add_Shown({ $form.Activate(); $inputBox.Focus() })
$dialogResult = $form.ShowDialog()
if ($dialogResult -ne [System.Windows.Forms.DialogResult]::OK) {
    [System.IO.File]::WriteAllText($statusPath, '{"status":"cancelled"}')
}
$form.Dispose()
