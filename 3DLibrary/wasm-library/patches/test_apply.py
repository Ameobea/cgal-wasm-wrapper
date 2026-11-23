#!/usr/bin/env python3
"""Exercise the queue against pristine headers from the pinned release archive."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / 'patches/manifest.json').read_text())
ARCHIVE = Path(os.environ.get('CGAL_TEST_ARCHIVE', ROOT / 'libraries/downloads' / MANIFEST['archive']['name']))


class PatchQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != MANIFEST['archive']['sha256']:
            raise ValueError('CGAL release archive checksum mismatch')
        with tarfile.open(ARCHIVE) as archive:
            cls.pristine = {name: archive.extractfile('CGAL-' + MANIFEST['cgal_version'] + '/' + name).read() for patch in MANIFEST['patches'] for name in patch['targets']}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test-cgal-patches-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'headers'
        for name, data in self.pristine.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def run_queue(self, *args, wrapper=ROOT):
        return subprocess.run([sys.executable, str(wrapper / 'apply-patches.py'), str(self.root), *args], capture_output=True, text=True)

    def snapshot(self):
        return {name: (self.root / name).read_bytes() for name in self.pristine}

    def assert_applied(self):
        for patch in MANIFEST['patches']:
            for name, hashes in patch['targets'].items():
                self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), hashes['after'])

    def test_apply_and_repeat(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_applied()
        before = self.snapshot()
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('already applied:'), len(MANIFEST['patches']))
        self.assertEqual(self.snapshot(), before)

    def test_dry_run_does_not_modify_headers(self):
        result = self.run_queue('--check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), self.pristine)

    def test_completes_partially_patched_tree(self):
        self.assertEqual(self.run_queue().returncode, 0)
        last = next(iter(MANIFEST['patches'][-1]['targets']))
        (self.root / last).write_bytes(self.pristine[last])
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('already applied:'), len(MANIFEST['patches']) - 1)
        self.assert_applied()

    def test_late_header_drift_prevents_all_writes(self):
        last = next(iter(MANIFEST['patches'][-1]['targets']))
        (self.root / last).write_bytes(self.pristine[last] + b'// drift\n')
        before = self.snapshot()
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.snapshot(), before)

    def bundle(self):
        bundle = Path(self.temp.name) / 'wrapper'
        shutil.copytree(ROOT / 'patches', bundle / 'patches')
        shutil.copyfile(ROOT / 'apply-patches.py', bundle / 'apply-patches.py')
        return bundle

    def test_patch_corruption_is_rejected(self):
        bundle = self.bundle()
        p = bundle / 'patches' / MANIFEST['patches'][0]['file']
        p.write_bytes(p.read_bytes() + b'corruption\n')
        self.assertNotEqual(self.run_queue(wrapper=bundle).returncode, 0)
        self.assertEqual(self.snapshot(), self.pristine)

    def test_wrong_output_hash_prevents_writes(self):
        bundle = self.bundle()
        manifest = json.loads((bundle / 'patches/manifest.json').read_text())
        next(iter(manifest['patches'][0]['targets'].values()))['after'] = '0' * 64
        (bundle / 'patches/manifest.json').write_text(json.dumps(manifest))
        self.assertNotEqual(self.run_queue(wrapper=bundle).returncode, 0)
        self.assertEqual(self.snapshot(), self.pristine)

    def test_selected_patch_only(self):
        result = self.run_queue('--only', MANIFEST['patches'][0]['file'])
        self.assertEqual(result.returncode, 0, result.stderr)
        first = next(iter(MANIFEST['patches'][0]['targets']))
        for name, data in self.snapshot().items():
            self.assertEqual(data == self.pristine[name], name != first)


if __name__ == '__main__':
    unittest.main()
