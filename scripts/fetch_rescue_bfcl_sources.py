"""Fetch only SHA-pinned official source files required by a frozen adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def fetch(manifests,output):
    output.mkdir(parents=True,exist_ok=True)
    for manifest in manifests:
        data=json.loads(manifest.read_text());commit=data['commit']
        if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Expected immutable official commit')
        for name,expected in data['files'].items():
            path=(output/name).resolve()
            if not path.is_relative_to(output.resolve()):raise ValueError('Source path escapes output root')
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Existing source differs; use a fresh output root: '+name)
                continue
            url=f'https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/{name}'
            with urllib.request.urlopen(url,timeout=30) as response:content=response.read()
            if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Remote source hash changed: '+name)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        target=output/manifest.name
        if target.exists() and target.read_bytes()!=manifest.read_bytes():raise ValueError('Existing manifest differs')
        if target.resolve()!=manifest.resolve():shutil.copy2(manifest,target)
    print('All pinned official sources verified')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();fetch(a.manifest,a.output)
