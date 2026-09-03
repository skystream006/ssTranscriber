import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_DIR / 'install_audio_tools.log'


def load_dotenv(path: Path):
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(REPO_ROOT / '.env')
_hf_token = (
    os.environ.get('HF_TOKEN')
    or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    or os.environ.get('HUGGINGFACE_HUB_TOKEN')
)
if _hf_token:
    os.environ.setdefault('HF_TOKEN', _hf_token)
    os.environ.setdefault('HUGGING_FACE_HUB_TOKEN', _hf_token)

CUDA_INDEX_URLS = {
    'cu118': 'https://download.pytorch.org/whl/cu118',
    'cu121': 'https://download.pytorch.org/whl/cu121',
    'cu124': 'https://download.pytorch.org/whl/cu124',
    'cu126': 'https://download.pytorch.org/whl/cu126',
    'cu130': 'https://download.pytorch.org/whl/cu130',
    'cpu': 'https://download.pytorch.org/whl/cpu',
}

parser = argparse.ArgumentParser(description='Install transcription dependencies.')
parser.add_argument(
    '--cuda',
    choices=sorted(CUDA_INDEX_URLS),
    default='cu124',
    help='PyTorch build to install. Use the CUDA version supported by your NVIDIA driver, or "cpu".',
)
parser.add_argument(
    '--force-reinstall',
    action='store_true',
    help='Force reinstall torch and torchaudio, even when the requested build is already installed.',
)
parser.add_argument(
    '--with-phowhisper',
    action='store_true',
    help='Validate a PhoWhisper model hosted on Hugging Face after installing dependencies.',
)
parser.add_argument(
    '--pho-model',
    default='vinai/PhoWhisper-large',
    help='PhoWhisper model ID on Hugging Face used with --with-phowhisper.',
)
parser.add_argument(
    '--with-parakeet',
    action='store_true',
    help='Install the NVIDIA NeMo toolkit and validate a Parakeet/Canary model on Hugging Face.',
)
parser.add_argument(
    '--parakeet-model',
    default='nvidia/parakeet-tdt-0.6b-v2',
    help='Parakeet or Canary model ID on Hugging Face used with --with-parakeet.',
)
parser.add_argument(
    '--with-sensevoice',
    action='store_true',
    help='Install FunASR and validate the SenseVoice model on Hugging Face.',
)
parser.add_argument(
    '--sensevoice-model',
    default='FunAudioLLM/SenseVoiceSmall',
    help='SenseVoice model ID on Hugging Face used with --with-sensevoice.',
)
parser.add_argument(
    '--with-viet-lyrics',
    action='store_true',
    help='Validate a kelvinbksoh Vietnamese lyrics-transcription Whisper model on Hugging Face.',
)
parser.add_argument(
    '--viet-lyrics-model',
    default='kelvinbksoh/whisper-large-v2-vietnamese-lyrics-transcription',
    help='Vietnamese lyrics-transcription model ID on Hugging Face used with --with-viet-lyrics.',
)
args = parser.parse_args()

if sys.version_info >= (3, 13) and args.cuda in {'cu118', 'cu121'}:
    parser.error(
        f'PyTorch does not provide Python {sys.version_info.major}.{sys.version_info.minor} wheels '
        f'on the {args.cuda} index. Use --cuda cu124, --cuda cu126, --cuda cu130, or --cuda cpu; '
        'otherwise run this installer with Python 3.12.'
    )

torch_command = [
    sys.executable, '-m', 'pip', 'install', '--upgrade',
    'torch', 'torchaudio', 'torchvision',
    '--index-url', CUDA_INDEX_URLS[args.cuda],
]
if args.force_reinstall:
    torch_command.insert(5, '--force-reinstall')

commands = [
    {
        'command': [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
        'required': True,
        'name': 'upgrade pip',
    },
    {
        'command': torch_command,
        'required': True,
        'name': 'install torch/torchaudio',
    },
    {
        'command': [
            sys.executable, '-m', 'pip', 'install', '--upgrade',
            'demucs', 'faster-whisper', 'transformers', 'sentencepiece', 'mutagen', 'huggingface_hub',
            'accelerate', 'librosa', 'soundfile',
            'nvidia-cublas-cu12==12.8.4.1', 'nvidia-cudnn-cu12==9.10.2.21',
        ],
        'required': True,
        'name': 'install audio dependencies',
    },
]
if args.with_phowhisper:
    phowhisper_verify_code = (
        'from huggingface_hub import model_info;'
        'from transformers import AutoConfig;'
        f'model={args.pho_model!r};'
        'info=model_info(model);'
        'cfg=AutoConfig.from_pretrained(model);'
        'print("phowhisper_model", model);'
        'print("repo_sha", getattr(info, "sha", None));'
        'print("model_type", getattr(cfg, "model_type", None));'
        'print("status", "ok")'
    )
    commands.append({
        'command': [sys.executable, '-c', phowhisper_verify_code],
        'required': True,
        'name': 'validate PhoWhisper model on Hugging Face',
    })
if args.with_parakeet:
    commands.append({
        'command': [
            sys.executable, '-m', 'pip', 'install', '--upgrade',
            'nemo_toolkit[asr]',
        ],
        'required': True,
        'name': 'install NVIDIA NeMo toolkit',
    })
    parakeet_verify_code = (
        'from huggingface_hub import model_info;'
        f'model={args.parakeet_model!r};'
        'info=model_info(model);'
        'print("parakeet_model", model);'
        'print("repo_sha", getattr(info, "sha", None));'
        'print("status", "ok")'
    )
    commands.append({
        'command': [sys.executable, '-c', parakeet_verify_code],
        'required': True,
        'name': 'validate Parakeet/Canary model on Hugging Face',
    })
if args.with_sensevoice:
    commands.append({
        'command': [
            sys.executable, '-m', 'pip', 'install', '--upgrade',
            'funasr', 'modelscope',
        ],
        'required': True,
        'name': 'install FunASR',
    })
    sensevoice_verify_code = (
        'from huggingface_hub import model_info;'
        f'model={args.sensevoice_model!r};'
        'info=model_info(model);'
        'print("sensevoice_model", model);'
        'print("repo_sha", getattr(info, "sha", None));'
        'print("status", "ok")'
    )
    commands.append({
        'command': [sys.executable, '-c', sensevoice_verify_code],
        'required': True,
        'name': 'validate SenseVoice model on Hugging Face',
    })
if args.with_viet_lyrics:
    # Uses the same transformers/accelerate/librosa/soundfile stack already
    # installed above, so no extra packages are needed here.
    viet_lyrics_verify_code = (
        'from huggingface_hub import model_info;'
        'from transformers import AutoConfig;'
        f'model={args.viet_lyrics_model!r};'
        'info=model_info(model);'
        'cfg=AutoConfig.from_pretrained(model);'
        'print("viet_lyrics_model", model);'
        'print("repo_sha", getattr(info, "sha", None));'
        'print("model_type", getattr(cfg, "model_type", None));'
        'print("status", "ok")'
    )
    commands.append({
        'command': [sys.executable, '-c', viet_lyrics_verify_code],
        'required': True,
        'name': 'validate viet-lyrics model on Hugging Face',
    })

verify_code = (
    'import torch;'
    'print("torch", torch.__version__);'
    'print("cuda_build", torch.version.cuda);'
    'print("cuda_available", torch.cuda.is_available());'
    'print("device_count", torch.cuda.device_count());'
    '[print("device", i, torch.cuda.get_device_name(i), "capability", torch.cuda.get_device_capability(i)) '
    'for i in range(torch.cuda.device_count())];'
    'torch.ones(1, device="cuda").add_(1).cpu() if torch.cuda.is_available() else None;'
    'print("cuda_kernel_test", "ok" if torch.cuda.is_available() else "skipped")'
)


def write_status(log, message):
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(line, flush=True)
    log.write(line + '\n')
    log.flush()


def run_and_tee(command, log):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
    return process.wait()


with open(log_path, 'w', encoding='utf-8') as log:
    write_status(log, f'Starting install (torch build: {args.cuda})')
    try:
        for step in commands:
            command = step['command']
            write_status(log, f'$ {" ".join(command)}')
            return_code = run_and_tee(command, log)
            write_status(log, f'EXIT_CODE={return_code}')
            if return_code != 0:
                print(f'Command failed: {" ".join(command)} (see {log_path})')
                sys.exit(return_code)

        write_status(log, '$ verify torch cuda')
        return_code = run_and_tee([sys.executable, '-c', verify_code], log)
        write_status(log, f'EXIT_CODE={return_code}')
        if return_code != 0:
            sys.exit(return_code)
    except Exception as e:
        write_status(log, f'EXCEPTION={e}')
        raise

print(f'Install complete. Log: {log_path}')
