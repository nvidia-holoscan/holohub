# License Addition Plan
**Created:** 2025-11-27 09:05:00
**Last Modified:** 2025-11-27 09:05:00
**By:** gemini-3-pro-preview

## Goal
Add the appropriate license file to the repository.

## Analysis
The user described the license as "commercially usable open source, but not MIT".
The `external/i4h-sensor-simulation` (NVIDIA Isaac for Healthcare) repository uses the **Apache License, Version 2.0**.
This license fits the description:
-   **Commercially usable**: Yes.
-   **Open Source**: Yes (OSI approved).
-   **Not MIT**: Yes (includes patent grants, more formal than MIT).

## Implementation Steps
1.  Create `LICENSE` file in the repository root.
2.  Content: Standard Apache License, Version 2.0 text (same as `external/i4h-sensor-simulation/LICENSE`).
3.  (Optional) Add `NOTICE` file if required (standard for Apache 2.0 to attribute dependencies).
4.  Update `README.md` to badge/mention the license.

## Verification
-   Ensure the text matches the standard Apache 2.0 license.


