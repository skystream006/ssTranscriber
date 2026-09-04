import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WEBUI_DIR = REPO_ROOT / 'webui'
REQUIREMENTS_PATH = REPO_ROOT / 'requirements-webui.txt'
ENV_PATH = REPO_ROOT / '.env'


def port_number(value: str):
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('port must be an integer') from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('port must be between 1 and 65535')
    return port


def read_env_value(key: str):
    if not ENV_PATH.is_file():
        return None
    for raw_line in ENV_PATH.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if line and not line.startswith('#') and '=' in line:
            name, _, value = line.partition('=')
            if name.strip() == key:
                return value.strip().strip('"').strip("'")
    return None


def write_env_value(key: str, value: str):
    lines = ENV_PATH.read_text(encoding='utf-8').splitlines() if ENV_PATH.is_file() else []
    replacement = f'{key}={value}'
    updated = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line and not line.startswith('#') and '=' in line and line.partition('=')[0].strip() == key:
            lines[index] = replacement
            updated = True
            break
    if not updated:
        if lines and lines[-1]:
            lines.append('')
        lines.append(replacement)
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_step(name: str, command: list[str], *, cwd: Path = REPO_ROOT):
    print(f'\n==> {name}')
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f'{name} failed with exit code {exc.returncode}.') from exc


def main():
    parser = argparse.ArgumentParser(description='Install and build the local Web API and React UI.')
    parser.add_argument(
        '--port',
        type=port_number,
        default=None,
        help='Persist the local Web API port in .env (default: existing value or 8000).',
    )
    args = parser.parse_args()

    configured_port = args.port
    if configured_port is None:
        configured_port = port_number(read_env_value('WEB_API_PORT') or '8000')
    else:
        write_env_value('WEB_API_PORT', str(configured_port))
        print(f'Configured WEB_API_PORT={configured_port} in .env')

    npm = shutil.which('npm.cmd') or shutil.which('npm')
    if npm is None:
        raise SystemExit('npm was not found on PATH. Install Node.js, then run this script again.')
    if not REQUIREMENTS_PATH.is_file():
        raise SystemExit(f'Missing dependency file: {REQUIREMENTS_PATH}')
    if not (WEBUI_DIR / 'package.json').is_file():
        raise SystemExit(f'Missing frontend project: {WEBUI_DIR / "package.json"}')

    run_step('Checking npm', [npm, '--version'])
    run_step(
        'Installing FastAPI dependencies',
        [sys.executable, '-m', 'pip', 'install', '-r', str(REQUIREMENTS_PATH)],
    )
    run_step('Installing React dependencies', [npm, 'install'], cwd=WEBUI_DIR)
    run_step('Building React production bundle', [npm, 'run', 'build'], cwd=WEBUI_DIR)

    print('\nWeb UI initialization complete.')
    print('Start it with: python .\\web_api.py')
    print(f'Then open: http://127.0.0.1:{configured_port}')


if __name__ == '__main__':
    main()