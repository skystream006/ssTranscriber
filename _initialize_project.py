from pathlib import Path
from shutil import copyfile

REPO_ROOT = Path(__file__).resolve().parent
DIRECTORIES = ('input', 'input/lyrics', 'output', 'temp')


def main():
    for directory_name in DIRECTORIES:
        directory = REPO_ROOT / directory_name
        existed = directory.is_dir()
        directory.mkdir(parents=True, exist_ok=True)
        print(f'{directory_name}/: {"already exists" if existed else "created"}')

    env_path = REPO_ROOT / '.env'
    if env_path.exists():
        print('.env: already exists')
    else:
        copyfile(REPO_ROOT / '.env.example', env_path)
        print('.env: created from .env.example')


if __name__ == '__main__':
    main()