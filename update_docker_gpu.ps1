[CmdletBinding()]
param(
	[switch]$SkipPull
)

$ErrorActionPreference = 'Stop'
$composeArguments = @('-f', 'compose.yaml', '-f', 'compose.gpu.yaml')

Push-Location $PSScriptRoot
try {
	if (-not $SkipPull) {
		git pull --ff-only
		if ($LASTEXITCODE -ne 0) {
			throw 'Git pull failed; the running container has not been changed.'
		}
	}

	docker compose --progress plain @composeArguments build sstranscriber
	if ($LASTEXITCODE -ne 0) {
		throw 'Docker build failed; the running container has not been changed.'
	}

	docker compose @composeArguments up --no-build -d sstranscriber
	if ($LASTEXITCODE -ne 0) {
		throw 'Docker container update failed.'
	}

	docker compose @composeArguments exec -T -u root sstranscriber sh -c 'if [ -d /usr/lib/wsl/drivers ]; then /sbin/ldconfig /usr/lib/wsl/drivers/*; fi'
	if ($LASTEXITCODE -ne 0) {
		throw 'GPU driver library setup failed.'
	}
} finally {
	Pop-Location
}