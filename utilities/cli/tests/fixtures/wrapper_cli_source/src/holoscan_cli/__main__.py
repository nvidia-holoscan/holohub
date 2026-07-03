# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

if sys.argv[1:] == ["build", "--help"]:
    pass
elif sys.argv[1:] == ["env-info"]:
    print(f"Executable: {sys.executable}")
else:
    raise SystemExit(2)
