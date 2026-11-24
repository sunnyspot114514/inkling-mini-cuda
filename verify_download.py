"""Verify this version of the Compact CUDA model and its optional prebuilt Windows kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def verify(folder,library):
    release=json.loads(Path(__file__).with_name("release-files.json").read_text(encoding="utf-8"))
    base=folder.resolve()
    manifest=base/"SHA256SUMS"
    if sha256(manifest)!=release["weights_manifest_sha256"]:
        raise ValueError("The model manifest differs from the one pinned by this source release")
    seen=set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected,name=line.split("  ",1)
        path=(base/name).resolve()
        if not re.fullmatch("[0-9a-f]{64}",expected) or not path.is_relative_to(base) or path in seen:
            raise ValueError("Unexpected entry in the model manifest")
        if sha256(path)!=expected:
            raise ValueError(f"Model checksum mismatch: {name}")
        seen.add(path)
    if len(seen)!=release["checksummed_files"]:
        raise ValueError("Unexpected model file count")
    print(f"Verified {len(seen)} files for {release['model_repo']} @ {release['version']}.")
    if library:
        actual=sha256(library)
        if actual!=release["windows_sm86_dll_sha256"]:
            raise ValueError("The DLL differs from the prebuilt binary shipped with this release")
        print(f"Verified Windows SM 8.6 DLL: {actual}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir",type=Path)
    p.add_argument("--release-dll",type=Path,help="Also check the prebuilt release DLL")
    args=p.parse_args()
    try:
        verify(args.model_dir,args.release_dll)
    except (OSError,ValueError) as err:
        p.exit(1,f"{err}\n")
