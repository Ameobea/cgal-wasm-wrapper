#!/usr/bin/env python3
"""Apply the pinned CGAL patch queue, verifying every input and output before writing."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply(root, only=(), check=False):
    patch_dir = HERE / 'patches'
    manifest = json.loads((patch_dir / 'manifest.json').read_text())
    known = {p['file'] for p in manifest['patches']}
    unknown = set(only) - known
    if unknown:
        raise ValueError(f'Unknown patches: {sorted(unknown)}')
    patches = [p for p in manifest['patches'] if not only or p['file'] in only]
    paths = {name for p in patches for name in p['targets']}
    originals = {name: (root / name).read_bytes() for name in paths}
    statuses = []
    with tempfile.TemporaryDirectory(prefix='cgal-patches-') as temp:
        stage = Path(temp)
        for name, data in originals.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for patch in patches:
            patch_path = patch_dir / patch['file']
            if digest(patch_path.read_bytes()) != patch['sha256']:
                raise ValueError(f"Patch checksum mismatch: {patch['file']}")
            current = {name: digest((stage / name).read_bytes()) for name in patch['targets']}
            if all(current[name] == hashes['after'] for name, hashes in patch['targets'].items()):
                statuses.append(('already applied', patch['file']))
                continue
            for name, hashes in patch['targets'].items():
                if current[name] != hashes['before']:
                    raise ValueError(f"Unexpected contents in {name} for {patch['file']}; expected the pinned upstream or fully patched file. No files changed.")
            subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1', '-d', str(stage), '-i', str(patch_path)], check=True, capture_output=True, text=True)
            for name, hashes in patch['targets'].items():
                if digest((stage / name).read_bytes()) != hashes['after']:
                    raise ValueError(f"Patched output checksum mismatch: {name}. No files changed.")
            statuses.append(('would apply' if check else 'applied', patch['file']))
        outputs = {name: (stage / name).read_bytes() for name in paths}
        if not check:
            # Validate the entire queue before touching any target, then detect concurrent edits.
            for name, data in originals.items():
                if (root / name).read_bytes() != data:
                    raise ValueError(f'File changed during preflight: {name}. No files changed.')
            for name, data in outputs.items():
                if data == originals[name]:
                    continue
                target = root / name
                mode = stat.S_IMODE(target.stat().st_mode)
                # Atomic per-file replacement. An OS failure can leave a prefix applied;
                # a later invocation safely recognizes those outputs by their hashes.
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.cgal-patch-', delete=False) as out:
                    pending = Path(out.name)
                    out.write(data)
                try:
                    pending.chmod(mode)
                    pending.replace(target)
                finally:
                    pending.unlink(missing_ok=True)
    for status, name in statuses:
        print(f'{status}: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=HERE / 'libraries/CGAL-6.2', help='CGAL release root (contains include/)')
    parser.add_argument('--only', action='append', default=[], metavar='PATCH', help='apply only this named patch; may be repeated')
    parser.add_argument('--check', action='store_true', help='validate/apply in temporary storage only')
    args = parser.parse_args()
    try:
        apply(args.root.resolve(), args.only, args.check)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
