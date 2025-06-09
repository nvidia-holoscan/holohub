import argparse
import os
import re
import shutil
from typing import List


def patch_file(file_path: str, script_dir: str) -> str | None:
    """Patch a single Python file. Returns backup path if patched, else None."""
    bench_import_lines = [
        "import sys\n",
        f'sys.path.append(r"{script_dir}")\n',
        "from benchmarked_application import BenchmarkedApplication\n",
    ]

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if any("BenchmarkedApplication" in line for line in lines):
        return None  # already patched

    backup = file_path + ".bak"
    shutil.copy2(file_path, backup)

    # insert after first import/from line.
    insert_idx = next(
        (i for i, line in enumerate(lines) if re.match(r"^\s*(import|from) ", line)), 0
    )
    patched_lines = lines[:insert_idx] + bench_import_lines + lines[insert_idx:]
    patched_lines = [
        line.replace("(Application)", "(BenchmarkedApplication)") for line in patched_lines
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(patched_lines)
    return backup


def patch_directory(root_dir: str, script_dir: str) -> List[str]:
    """Patch all python files under root_dir. Returns list of backup paths."""
    backups: List[str] = []
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".py"):
                fp = os.path.join(dirpath, fname)
                bak = patch_file(fp, script_dir)
                if bak:
                    backups.append(bak)
    return backups


def restore_backups(backups: List[str]):
    for bak in backups:
        if bak.endswith(".bak"):
            orig = bak[:-4]
            if os.path.exists(bak):
                shutil.move(bak, orig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Patch Holohub Python application sources for benchmarking."
    )
    parser.add_argument("target_dir", help="Root directory to recurse and patch")
    args = parser.parse_args()

    script_directory = os.path.dirname(os.path.abspath(__file__))
    patched = patch_directory(args.target_dir, script_directory)
    print(f"Patched {len(patched)} python files")
