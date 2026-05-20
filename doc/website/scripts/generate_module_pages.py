# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Generate Holoscan Modules website pages from module-sites.json.

Runs as a mkdocs gen-files plugin script. For each module in modules/module-sites.json:
  - In-tree modules: reads metadata.json and README from the HoloHub tree directly.
  - External modules: shallow git-clones the repo at the pinned ref, reads those files.

Writes per-module detail .md pages via mkdocs_gen_files, and writes the card/nav HTML
fragments to overrides/_pages/ via direct file I/O (same pattern as generate_featured_apps.py).
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import mkdocs_gen_files

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from clone_module import clone_external_module  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HOLOHUB_GITHUB_BASE = "https://github.com/nvidia-holoscan/holohub"
HOLOHUB_RAW_BASE = "https://raw.githubusercontent.com/nvidia-holoscan/holohub/main"

QUALITY_SCORE_LABELS = {
    1: "Basic",
    2: "Developing",
    3: "Trusted",
    4: "Reliable",
    5: "Excellent",
}

QUALITY_SCORE_COLORS = {
    1: "#888888",
    2: "#aaa000",
    3: "#5f9300",
    4: "#3a7d44",
    5: "#1a5c2e",
}


# ---------------------------------------------------------------------------
# Repository root resolution
# ---------------------------------------------------------------------------


def get_holohub_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    return Path(result.stdout.strip())


# ---------------------------------------------------------------------------
# module-sites.json loading
# ---------------------------------------------------------------------------


def load_module_sites(holohub_root: Path) -> list[dict]:
    sites_path = holohub_root / "modules" / "module-sites.json"
    with sites_path.open() as f:
        data = json.load(f)
    return data.get("modules", [])


# ---------------------------------------------------------------------------
# Metadata resolution
# ---------------------------------------------------------------------------


def resolve_module_metadata(
    entry: dict, holohub_root: Path
) -> tuple[dict, Path, object] | None:
    """Return (full_metadata_dict, module_root, tmp_dir_or_None).

    module_root is the directory that contains metadata.json.
    tmp_dir is a TemporaryDirectory for external clones; caller must call .cleanup().
    Returns None and logs a warning on any failure.
    """
    name = entry["name"]
    url = entry.get("url")
    ref = entry.get("ref")

    if not url:
        # In-tree module
        module_root = holohub_root / "modules" / name
        metadata_path = module_root / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"In-tree module '{name}': metadata.json not found at {metadata_path}")
            return None
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
            return metadata, module_root, None
        except Exception as e:
            logger.warning(f"Failed to read metadata for in-tree module '{name}': {e}")
            return None
    else:
        # Skip placeholder URLs (used during initial setup before real URLs are known)
        if "placeholder" in url:
            logger.warning(
                f"Skipping external module '{name}': URL '{url}' is a placeholder. "
                "Update module-sites.json with the real repository URL."
            )
            return None

        # External module — shallow clone
        try:
            clone_path, tmp_dir = clone_external_module(url, ref)
            metadata_path = clone_path / "metadata.json"
            if not metadata_path.exists():
                logger.warning(
                    f"External module '{name}': no metadata.json found at repo root after cloning {url}@{ref}"
                )
                tmp_dir.cleanup()
                return None
            with metadata_path.open() as f:
                metadata = json.load(f)
            return metadata, clone_path, tmp_dir
        except Exception as e:
            logger.warning(f"Failed to clone external module '{name}' from {url}@{ref}: {e}")
            return None


# ---------------------------------------------------------------------------
# README resolution
# ---------------------------------------------------------------------------


def resolve_readme(metadata_module: dict, module_root: Path) -> str:
    """Read the README for a module. Returns raw Markdown or a stub."""
    doc_block = metadata_module.get("documentation", {})
    readme_rel = doc_block.get("readme") if doc_block else None

    if readme_rel:
        readme_path = (module_root / readme_rel).resolve()
    else:
        readme_path = module_root / "README.md"

    if readme_path.exists():
        try:
            return readme_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not read README at {readme_path}: {e}")

    logger.warning(f"No README found for module '{metadata_module.get('name')}' at {readme_path}")
    return f"# {metadata_module.get('name', 'Module')}\n\n*No README available.*\n"


# ---------------------------------------------------------------------------
# Image URL patching
# ---------------------------------------------------------------------------

_IMG_PATTERN = re.compile(
    r'(!\[[^\]]*\]\()([^)]+)(\))|(<img\s[^>]*src=")([^"]+)(")',
    re.IGNORECASE,
)


def _rewrite_image_url(raw_url: str, module_root: Path, source_url: str, ref: str) -> str:
    """Rewrite a single image URL to an absolute raw GitHub URL."""
    if raw_url.startswith(("http://", "https://", "data:")):
        return raw_url  # already absolute

    # Determine raw base URL
    if source_url.startswith(HOLOHUB_GITHUB_BASE):
        # In-tree: resolve relative to module_root against holohub tree
        try:
            resolved = (module_root / raw_url).resolve()
            holohub_root = Path(
                subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            rel = resolved.relative_to(holohub_root)
            return f"{HOLOHUB_RAW_BASE}/{rel}"
        except Exception:
            return raw_url
    else:
        # External: rewrite relative to repo root via raw URL
        clean_url = source_url.rstrip("/")
        # Convert github.com/org/repo to raw.githubusercontent.com/org/repo
        raw_base = re.sub(
            r"^https://github\.com/", "https://raw.githubusercontent.com/", clean_url
        )
        return f"{raw_base}/{ref}/{raw_url.lstrip('/')}"


def patch_readme_images(readme_text: str, module_root: Path, source_url: str, ref: str) -> str:
    """Rewrite relative image paths in README Markdown to absolute raw GitHub URLs."""

    def replacer(m: re.Match) -> str:
        if m.group(1):  # Markdown image ![alt](url)
            new_url = _rewrite_image_url(m.group(2), module_root, source_url, ref)
            return f"{m.group(1)}{new_url}{m.group(3)}"
        else:  # HTML <img src="url">
            new_url = _rewrite_image_url(m.group(5), module_root, source_url, ref)
            return f"{m.group(4)}{new_url}{m.group(6)}"

    return _IMG_PATTERN.sub(replacer, readme_text)


# ---------------------------------------------------------------------------
# Metadata header (octicons-decorated block)
# ---------------------------------------------------------------------------


def build_metadata_header(metadata_module: dict, entry: dict) -> str:
    """Generate the octicons metadata block for a module detail page."""
    authors = metadata_module.get("authors", [])
    authors_str = ", ".join(
        f'{a.get("name", "")} ({a.get("affiliation", "")})' for a in authors
    ) or None

    platforms = metadata_module.get("platforms", [])
    platforms_str = ", ".join(platforms) if platforms else None

    language = metadata_module.get("language", [])
    language_str = ", ".join(language) if isinstance(language, list) else language or None

    version = metadata_module.get("version")

    hsdk = metadata_module.get("holoscan_sdk", {})
    min_sdk = hsdk.get("minimum_required_version") if hsdk else None
    tested_sdks = hsdk.get("tested_versions", []) if hsdk else []
    tested_sdk_str = ", ".join(tested_sdks) if tested_sdks else None

    license_str = metadata_module.get("license")

    score = entry.get("nvidia_quality_score", 0)
    quality_str = QUALITY_SCORE_LABELS.get(score, str(score))

    binary_pkgs = metadata_module.get("binary_packages", {})
    install_cmds = binary_pkgs.get("install_commands", []) if binary_pkgs else []
    install_str = " · ".join(f"`{cmd}`" for cmd in install_cmds) if install_cmds else None

    homepage = metadata_module.get("homepage")
    homepage_str = f"[{homepage}]({homepage})" if homepage else None

    lines_input = [
        ("tag", "Version", version),
        ("person", "Authors", authors_str),
        ("device-desktop", "Supported platforms", platforms_str),
        ("code-square", "Language", language_str),
        ("stack", "Minimum Holoscan SDK version", min_sdk),
        ("beaker", "Tested Holoscan SDK versions", tested_sdk_str),
        ("law", "License", license_str),
        ("sparkle-fill", "NVIDIA quality score", quality_str),
        ("package", "Install", install_str),
        ("home", "Homepage", homepage_str),
    ]

    output_lines = []
    for icon, label, value in lines_input:
        if not value:
            continue
        output_lines.append(f":octicons-{icon}-24: **{label}:** {value}<br>")

    return "".join(output_lines) + "<br>"


# ---------------------------------------------------------------------------
# Detail page generation
# ---------------------------------------------------------------------------


def generate_detail_page(
    entry: dict,
    metadata_module: dict,
    readme_text: str,
    source_url: str,
    ref: str | None,
) -> None:
    """Write a per-module detail page to the mkdocs virtual filesystem."""
    name = entry["name"]
    tags = metadata_module.get("tags", [])
    tags_yaml = "\n".join(f"  - {t}" for t in tags)

    frontmatter = f'---\ntitle: "{name}"\ntags:\n{tags_yaml}\n---\n\n'

    metadata_header = build_metadata_header(metadata_module, entry)

    # Patch the first H1 to link to the source URL
    readme_body = readme_text
    h1_match = re.match(r"(# .+?)(\n|$)", readme_body)
    if h1_match and source_url:
        original_h1 = h1_match.group(0)
        h1_text = original_h1.lstrip("# ").strip()
        linked_h1 = f"# [{h1_text}]({source_url})\n"
        readme_body = readme_body.replace(original_h1, linked_h1, 1)

    # Insert metadata header after the first H1
    first_newline = readme_body.find("\n")
    if first_newline != -1:
        readme_body = readme_body[: first_newline + 1] + "\n" + metadata_header + "\n" + readme_body[first_newline + 1 :]
    else:
        readme_body = readme_body + "\n\n" + metadata_header

    footer = f"\n---\n[View source on GitHub]({source_url})\n" if source_url else ""

    page_content = frontmatter + readme_body + footer

    with mkdocs_gen_files.open(f"modules/{name}.md", "w") as f:
        f.write(page_content)

    logger.info(f"Generated detail page for module '{name}'")


# ---------------------------------------------------------------------------
# Card HTML generation
# ---------------------------------------------------------------------------


def generate_module_card(entry: dict, metadata_module: dict) -> str:
    """Return one card HTML block for the modules index page."""
    name = entry["name"]
    url = entry.get("url")
    score = entry.get("nvidia_quality_score", 0)
    version = metadata_module.get("version", "")
    description = metadata_module.get("description", "")
    operators = metadata_module.get("operator_names", [])
    platforms = metadata_module.get("platforms", [])
    language = metadata_module.get("language", [])
    license_str = metadata_module.get("license", "")
    tags = metadata_module.get("tags", [])

    # First tag drives sidebar filter
    primary_tag = tags[0] if tags else "Other"

    source_badge_color = "#5f9300" if not url else "#0077b6"
    source_badge_label = "In-Tree" if not url else "External"

    operator_badges = "".join(
        f'<span style="display:inline-block;background:var(--md-code-bg-color);'
        f'color:var(--md-code-fg-color);padding:0.1rem 0.4rem;border-radius:0.2rem;'
        f'font-size:0.6rem;margin:0.1rem;">{op}</span>'
        for op in operators
    )

    platform_pills = "".join(
        f'<span style="display:inline-block;background:var(--md-default-fg-color--lightest);'
        f'color:var(--md-default-fg-color);padding:0.1rem 0.4rem;border-radius:0.2rem;'
        f'font-size:0.6rem;margin:0.1rem;">{p}</span>'
        for p in platforms
    )

    lang_badges = "".join(
        f'<span style="display:inline-block;background:#1a73e8;color:white;'
        f'padding:0.1rem 0.4rem;border-radius:0.2rem;font-size:0.6rem;margin:0.1rem;">{lang}</span>'
        for lang in language
    )

    quality_label = QUALITY_SCORE_LABELS.get(score, str(score))
    quality_color = QUALITY_SCORE_COLORS.get(score, "#888888")
    quality_badge_html = (
        f'<span style="background:{quality_color};color:white;padding:0.15rem 0.4rem;'
        f'border-radius:0.2rem;font-size:0.6rem;font-weight:600;" '
        f'title="NVIDIA quality score: {score}/5">{quality_label}</span>'
    )

    detail_url = f"{name}/"

    return f"""<div class="col-xl-4 col-lg-6 col-sm-12 mb-1 feature-box">
  <span class="md-tag" style="display:none;">{primary_tag}</span>
  <a class="app-card shadow padding-feature-box-item bg-white d-block" href="{detail_url}" style="text-decoration:none;color:inherit;position:relative;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.3rem;">
      <h3 style="margin:0;font-size:0.95rem;">{name}</h3>
      <div style="display:flex;gap:0.3rem;flex-shrink:0;">
        {f'<span style="background:#2a9d8f;color:white;padding:0.15rem 0.4rem;border-radius:0.2rem;font-size:0.6rem;font-weight:600;">v{version}</span>' if version else ''}
        <span style="background:{source_badge_color};color:white;padding:0.15rem 0.4rem;border-radius:0.2rem;font-size:0.6rem;font-weight:600;">{source_badge_label}</span>
      </div>
    </div>
    <p class="feature-card-desc" style="font-size:0.72rem;color:var(--md-default-fg-color--light);margin-bottom:0.5rem;min-height:60px;">{description}</p>
    <div style="margin-bottom:0.4rem;">
      <span style="font-size:0.6rem;font-weight:600;color:var(--md-default-fg-color--light);">Operators: </span>
      {operator_badges if operator_badges else '<span style="font-size:0.6rem;color:var(--md-default-fg-color--lighter);">—</span>'}
    </div>
    <div style="margin-bottom:0.4rem;">
      {lang_badges}
      {platform_pills}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;">
      <div>
        {f'<span style="font-size:0.6rem;color:var(--md-default-fg-color--light);">{license_str}</span>' if license_str else ''}
        <span style="margin-left:0.5rem;">{quality_badge_html}</span>
      </div>
      <span class="nv-teaser-text-link" style="font-size:0.65rem;color:#76b900;">View Details &rsaquo;</span>
    </div>
  </a>
</div>"""


# ---------------------------------------------------------------------------
# _pages HTML writers (real filesystem, not mkdocs_gen_files)
# ---------------------------------------------------------------------------


def write_modules_cards_html(cards: list[str], overrides_pages_dir: Path) -> None:
    overrides_pages_dir.mkdir(parents=True, exist_ok=True)
    out_path = overrides_pages_dir / "modules.html"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(cards) + "\n")
    logger.info(f"Wrote {len(cards)} module cards to {out_path}")


def write_modules_nav_html(tag_counts: dict, total_count: int, overrides_pages_dir: Path) -> None:
    overrides_pages_dir.mkdir(parents=True, exist_ok=True)
    out_path = overrides_pages_dir / "modules_nav.html"

    nav_style = (
        'style="display:block;padding:0.5rem 1rem;color:var(--md-default-fg-color);'
        'text-decoration:none;font-size:0.8rem;border-left:3px solid transparent;"'
    )

    lines = [
        f'<a href="#all" onclick="filterByTag(\'all\'); return true;" {nav_style}>All ({total_count})</a>'
    ]
    for tag, count in sorted(tag_counts.items()):
        safe_tag = tag.lower().replace(" ", "-")
        lines.append(
            f'<a href="#{safe_tag}" onclick="filterByTag(\'{tag}\'); return true;" {nav_style}>'
            f"{tag} ({count})</a>"
        )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Wrote modules nav HTML to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    holohub_root = get_holohub_root()
    overrides_pages_dir = Path(__file__).resolve().parent.parent / "overrides" / "_pages"

    entries = load_module_sites(holohub_root)
    logger.info(f"Found {len(entries)} module(s) in module-sites.json")

    cards: list[str] = []
    tag_counts: dict[str, int] = {}

    for entry in entries:
        name = entry.get("name", "<unnamed>")
        url = entry.get("url")
        ref = entry.get("ref", "main")

        result = resolve_module_metadata(entry, holohub_root)
        if result is None:
            logger.warning(f"Skipping module '{name}' — could not resolve metadata")
            continue

        metadata, module_root, tmp_dir = result

        try:
            # Unwrap top-level "module" key from schema v2
            metadata_module = metadata.get("module", metadata)

            # Determine source URL for links and image patching.
            # source_url in module-sites.json overrides the default for in-tree modules
            # that represent external projects (decouples linking from cloning).
            source_url = (
                entry.get("source_url")
                or (url.rstrip("/") if url else f"{HOLOHUB_GITHUB_BASE}/tree/main/modules/{name}")
            )

            readme_text = resolve_readme(metadata_module, module_root)
            readme_text = patch_readme_images(readme_text, module_root, source_url, ref)

            generate_detail_page(entry, metadata_module, readme_text, source_url, ref)

            card_html = generate_module_card(entry, metadata_module)
            cards.append(card_html)

            # Accumulate tag counts for sidebar nav (use first tag as category)
            tags = metadata_module.get("tags", [])
            if tags:
                primary_tag = tags[0]
                tag_counts[primary_tag] = tag_counts.get(primary_tag, 0) + 1

        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    write_modules_cards_html(cards, overrides_pages_dir)
    write_modules_nav_html(tag_counts, len(cards), overrides_pages_dir)

    logger.info(f"Module page generation complete: {len(cards)} module(s) rendered")


if __name__ in {"__main__", "<run_path>"}:
    main()
