# -*- coding: utf-8 -*-
#
# (C) Copyright 2022 Karellen, Inc. (https://www.karellen.co/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import csv
import os
from typing import Tuple, List

from pip._internal.models.scheme import SCHEME_KEYS
from pip._internal.operations.install.wheel import (
    _fs_to_record_path,
    is_within_directory,
    chain,
    InstallationError
)

from wheel_axle.runtime._common import Installer
from wheel_axle.runtime.constants import SYMLINKS_FILE

Symlink = Tuple[str, str, bool]
Symlinks = List[Symlink]


def write_symlinks_file(symlinks_file: str, symlinks: Symlinks) -> None:
    with open(symlinks_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows([symlink[0].replace(os.sep, "/"),
                          symlink[1].replace(os.sep, "/"),
                          int(symlink[2])]
                         for symlink in symlinks)


def read_symlinks_file(symlinks_file: str) -> Symlinks:
    with open(symlinks_file, "r") as f:
        reader = csv.reader(f)
        results = []
        for row in reader:
            results.append((row[0].replace("/", os.sep),
                            row[1].replace("/", os.sep),
                            bool(int(row[2]))))
        return results


class SymlinksInstaller(Installer):
    def install(self) -> None:
        symlinks_path = os.path.join(self.dist_info_dir, SYMLINKS_FILE)

        if not os.path.exists(symlinks_path):
            return

        symlinks = read_symlinks_file(symlinks_path)
        self._install_symlinks(symlinks)

    def _install_symlinks(self, symlinks: Symlinks):
        scheme_paths = {key: getattr(self.scheme, key) for key in SCHEME_KEYS}
        symlinking_paths = list(set(chain((self.lib_dir,), scheme_paths.values())))
        real_symlinking_paths = list(map(os.path.realpath, symlinking_paths))

        def assert_no_path_traversal(dest_dir_path: str, target_path: str) -> None:
            if not is_within_directory(dest_dir_path, target_path):
                message = (
                    "The distribution {!r} has a file {!r} trying to install"
                    " outside the target directory {!r}"
                )
                raise InstallationError(
                    message.format(self.dist_meta.project_name, target_path, dest_dir_path)
                )

        def assert_no_path_symlinking(symlink_path: str, symlink_target: str) -> None:
            for real_symlinking_path in real_symlinking_paths:
                if is_within_directory(real_symlinking_path, symlink_target):
                    return

            message = (
                "The distribution {!r} has a symlink {!r} trying to link to {!r}"
                " outside the allowed target directories {!r}"
            )
            raise InstallationError(
                message.format(self.dist_meta.project_name, symlink_path, symlink_target,
                               ",".join(real_symlinking_paths))
            )

        def make_root_scheme_path(dest: str, record_path: str) -> Tuple[str, str]:
            normed_path = os.path.normpath(record_path)
            dest_path = os.path.join(dest, normed_path)
            assert_no_path_traversal(dest, dest_path)
            return dest, dest_path

        def make_data_scheme_path(record_path: str) -> Tuple[str, str]:
            normed_path = os.path.normpath(record_path)
            try:
                _, scheme_key, dest_subpath = normed_path.split(os.path.sep, 2)
            except ValueError:
                message = (
                    "Unexpected file in {}: {!r}. .data directory contents"
                    " should be named like: '<scheme key>/<path>'."
                ).format(self.dist_meta.project_name, record_path)
                raise InstallationError(message)

            try:
                scheme_path = scheme_paths[scheme_key]
            except KeyError:
                valid_scheme_keys = ", ".join(sorted(scheme_paths))
                message = (
                    "Unknown scheme key used in {}: {} (for file {!r}). .data"
                    " directory contents should be in subdirectories named"
                    " with a valid scheme key ({})"
                ).format(self.dist_meta.project_name, scheme_key, record_path, valid_scheme_keys)
                raise InstallationError(message)

            dest_path = os.path.join(scheme_path, dest_subpath)
            assert_no_path_traversal(scheme_path, dest_path)
            return scheme_path, dest_path

        def is_data_scheme_path(path: str) -> bool:
            return path.split("/", 1)[0].endswith(".data")

        for symlink in symlinks:
            symlink_path, symlink_target, is_symlink_dir = symlink
            if is_data_scheme_path(symlink_path):
                # Data scheme
                scheme_path, norm_symlink_path = make_data_scheme_path(symlink_path)
            else:
                # Root scheme
                scheme_path, norm_symlink_path = make_root_scheme_path(self.lib_dir, symlink_path)

            # This has to be done in order one by one, because previously created symlinks
            # will affect normpath resolution
            real_symlink_target = os.path.realpath(
                os.path.join(os.path.dirname(norm_symlink_path), symlink_target))

            # Here we will compare real paths
            assert_no_path_symlinking(norm_symlink_path, real_symlink_target)

            # os.path.exists doesn't work with broken symlinks, returns False
            # so we check if something is link or it otherwise exists
            if os.path.islink(norm_symlink_path) or os.path.exists(norm_symlink_path):
                os.unlink(norm_symlink_path)

            # We preserve symlink target as de-normalized, while checking traversal as real
            os.symlink(symlink_target, norm_symlink_path, is_symlink_dir)
            self.record_installed(norm_symlink_path, norm_symlink_path, False)

    def record_installed(self,
                         srcfile: str,
                         destfile: str,
                         modified: bool = False
                         ) -> None:
        """Map archive RECORD paths to installation RECORD paths."""
        newpath = _fs_to_record_path(destfile, self.lib_dir)
        self.installed[newpath] = newpath
        # if modified:
        #    self.changed.add(_fs_to_record_path(destfile))
