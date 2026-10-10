"""Download official SOLD SAVi weights and verify the Git LFS SHA-256."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

CHECKPOINTS = {
    'reach_red': '9c937fee8edb9f5c9431cec3b4b911f28fd530242cc273d0a2ef90c3525031d8',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=CHECKPOINTS, default='reach_red')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / 'checkpoints/savi' / f'{args.task}.ckpt'
    expected = CHECKPOINTS[args.task]
    if output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest() == expected:
        print(f'Already verified: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    url = ('https://media.githubusercontent.com/media/maltemosbach/sold/master/'
           f'checkpoints/autoencoder/savi/{args.task}/latest.ckpt')
    # Verify before replacing any existing file.
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            stream.close()
            actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'SOLD checkpoint checksum mismatch: {actual}')
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Verified: {output} (SHA-256 {expected})')


if __name__ == '__main__':
    main()
