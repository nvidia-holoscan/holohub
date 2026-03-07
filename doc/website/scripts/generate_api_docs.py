# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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
"""Generate API reference documentation for HoloHub operators.

This module generates Markdown API reference content for C++ and Python operators
by processing Doxygen XML output (C++) and parsing Python source files / pydoc headers.

The generated content is stored in a dictionary keyed by operator directory path,
and consumed by generate_pages.py to append API Reference sections to operator pages.
"""

import ast
import logging
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Doxygen XML parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def _xml_text(element) -> str:
    """Recursively extract all text content from an XML element."""
    if element is None:
        return ""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_xml_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _parse_param_type(node) -> str:
    """Extract a parameter type string from a Doxygen <type> node."""
    if node is None:
        return ""
    return _xml_text(node).replace("< ", "<").replace(" >", ">")


def _parse_member_def(member) -> dict | None:
    """Parse a Doxygen <memberdef> element into a dict of method/field info."""
    kind = member.get("kind")
    if kind not in ("function", "variable"):
        return None

    prot = member.get("prot", "public")
    if prot != "public":
        return None

    name = _xml_text(member.find("name"))
    # Skip compiler-generated, internal members, and destructors
    if not name or name.startswith("_") or name.startswith("~"):
        return None

    # Skip default constructors (not useful in API docs)
    argsstring = _xml_text(member.find("argsstring"))
    if kind == "function" and "=default" in argsstring.replace(" ", ""):
        return None

    brief = _xml_text(member.find("briefdescription"))
    detailed = _xml_text(member.find("detaileddescription"))
    description = brief or detailed

    result = {
        "kind": kind,
        "name": name,
        "description": description,
        "prot": prot,
    }

    if kind == "function":
        return_type = _parse_param_type(member.find("type"))
        result["return_type"] = return_type
        argsstring = _xml_text(member.find("argsstring"))
        result["argsstring"] = argsstring

        params = []
        for param in member.findall("param"):
            ptype = _parse_param_type(param.find("type"))
            pname = _xml_text(param.find("declname"))
            params.append({"type": ptype, "name": pname})
        result["params"] = params

    elif kind == "variable":
        var_type = _parse_param_type(member.find("type"))
        result["type"] = var_type

    return result


def _parse_compound_file(xml_path: Path) -> dict | None:
    """Parse a Doxygen compound XML file for a single class.

    Returns a dict with class info or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        logger.warning(f"Failed to parse XML: {xml_path}")
        return None

    root = tree.getroot()
    compounddef = root.find("compounddef")
    if compounddef is None:
        return None

    kind = compounddef.get("kind")
    if kind != "class":
        return None

    compound_name = _xml_text(compounddef.find("compoundname"))
    if not compound_name:
        return None

    # Get location info to map back to source file
    location = compounddef.find("location")
    header_file = location.get("file", "") if location is not None else ""

    # Get base classes
    base_classes = []
    for base in compounddef.findall("basecompoundref"):
        base_classes.append(_xml_text(base))

    brief = _xml_text(compounddef.find("briefdescription"))
    detailed = _xml_text(compounddef.find("detaileddescription"))
    description = brief or detailed

    # Parse members by section
    methods = []
    parameters = []

    for section in compounddef.findall("sectiondef"):
        for member in section.findall("memberdef"):
            parsed = _parse_member_def(member)
            if parsed is None:
                continue

            if parsed["kind"] == "function":
                methods.append(parsed)
            elif parsed["kind"] == "variable":
                # Holoscan Parameter<T> types are operator parameters
                if "Parameter<" in parsed.get("type", ""):
                    param_type = parsed["type"]
                    # Strip the outer Parameter<...> wrapper. Using string slicing
                    # rather than a regex avoids truncation for nested template types
                    # like Parameter<std::vector<int>> (a lazy regex would stop at
                    # the first ">").
                    if param_type.startswith("Parameter<") and param_type.endswith(">"):
                        inner_type = param_type[len("Parameter<") : -1].strip()
                    else:
                        inner_type = param_type
                    parsed["type"] = inner_type
                    parameters.append(parsed)

    return {
        "name": compound_name,
        "header_file": header_file,
        "base_classes": base_classes,
        "description": description,
        "methods": methods,
        "parameters": parameters,
    }


def run_doxygen_and_parse(website_dir: Path) -> dict:
    """Run Doxygen and parse resulting XML into a mapping of header path -> class info.

    Returns:
        Dict mapping header file paths (relative to repo root) to lists of class info dicts.
    """
    doxyfile = website_dir / "Doxyfile"
    if not doxyfile.exists():
        logger.error(f"Doxyfile not found at {doxyfile}")
        return {}

    # Check for doxygen
    if not shutil.which("doxygen"):
        logger.warning("Doxygen not found in PATH. Skipping C++ API doc generation.")
        return {}

    xml_output_dir = website_dir / "_build" / "doxygen" / "xml"

    # Ensure output directory exists
    (website_dir / "_build" / "doxygen").mkdir(parents=True, exist_ok=True)

    # Run doxygen
    logger.info("Running Doxygen on operator headers...")
    try:
        result = subprocess.run(
            ["doxygen", str(doxyfile)],
            cwd=str(website_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        logger.error("Doxygen timed out after 300 seconds. Skipping C++ API doc generation.")
        return {}
    if result.returncode != 0:
        logger.error(f"Doxygen failed: {result.stderr}")
        return {}
    logger.info("Doxygen completed successfully.")

    if not xml_output_dir.exists():
        logger.error(f"Doxygen XML output not found at {xml_output_dir}")
        return {}

    # Parse the index to find all class compound files
    index_path = xml_output_dir / "index.xml"
    if not index_path.exists():
        logger.error("Doxygen index.xml not found")
        return {}

    try:
        index_tree = ET.parse(index_path)
    except ET.ParseError:
        logger.error("Failed to parse Doxygen index.xml")
        return {}

    # Build mapping: header_file -> [class_info, ...]
    header_to_classes = {}

    for compound in index_tree.getroot().findall("compound"):
        if compound.get("kind") != "class":
            continue

        refid = compound.get("refid")
        compound_xml = xml_output_dir / f"{refid}.xml"
        if not compound_xml.exists():
            continue

        class_info = _parse_compound_file(compound_xml)
        if class_info is None:
            continue

        header = class_info["header_file"]
        if header:
            header_to_classes.setdefault(header, []).append(class_info)

    logger.info(
        f"Parsed {sum(len(v) for v in header_to_classes.values())} C++ classes from Doxygen XML"
    )
    return header_to_classes


# ──────────────────────────────────────────────────────────────────────────────
# Python source parsing (pure Python operators)
# ──────────────────────────────────────────────────────────────────────────────


def _parse_python_source(py_path: Path) -> list[dict]:
    """Parse a Python source file using AST to extract operator class information.

    Returns a list of class info dicts for classes that inherit from Operator.
    """
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse Python source {py_path}: {e}")
        return []

    classes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if any base class might be Operator
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(ast.unparse(base))

        # Include classes that inherit from Operator or a base class ending in "Op".
        # Using "Operator" substring and endswith("Op") avoids false positives from
        # class names like "Option" or "ConfigOption" that contain "Op" as a substring.
        is_operator = any("Operator" in b or b.endswith("Op") for b in base_names)
        if not is_operator:
            continue

        class_doc = ast.get_docstring(node) or ""

        methods = []
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Skip private methods except __init__
            if item.name.startswith("_") and item.name != "__init__":
                continue

            method_doc = ast.get_docstring(item) or ""

            # Build argument string
            args = []
            for arg in item.args.args:
                if arg.arg == "self":
                    continue
                ann = ast.unparse(arg.annotation) if arg.annotation else ""
                args.append({"name": arg.arg, "type": ann})

            methods.append(
                {
                    "name": item.name,
                    "description": method_doc.split("\n")[0] if method_doc else "",
                    "full_docstring": method_doc,
                    "params": args,
                }
            )

        # Extract class-level type annotations (operator parameters)
        parameters = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                param_name = item.target.id
                param_type = ast.unparse(item.annotation) if item.annotation else ""
                default = ast.unparse(item.value) if item.value else ""
                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "default": default,
                    }
                )

        classes.append(
            {
                "name": node.name,
                "base_classes": base_names,
                "description": class_doc.split("\n\n")[0] if class_doc else "",
                "full_docstring": class_doc,
                "methods": methods,
                "parameters": parameters,
                "source_file": str(py_path),
            }
        )

    return classes


# ──────────────────────────────────────────────────────────────────────────────
# Pydoc.hpp parsing (pybind11 operators)
# ──────────────────────────────────────────────────────────────────────────────


def _parse_pydoc_hpp(pydoc_path: Path) -> list[dict]:
    """Parse a *_pydoc.hpp file to extract Python docstrings for pybind11 operators.

    Returns a list of dicts, one per inner namespace (operator class). A single
    pydoc.hpp file may define multiple operator classes in separate namespaces
    (e.g. iio_controller_pydoc.hpp defines five). Each dict contains the class
    name and its method docstrings.
    """
    try:
        content = pydoc_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    # Extract PYDOC entries: PYDOC(name, R"doc(...)doc")
    pydoc_pattern = re.compile(
        r'PYDOC\(\s*(\w+)\s*,\s*R"doc\((.*?)\)doc"\s*\)',
        re.DOTALL,
    )

    def _make_record(class_name: str, docs: dict) -> dict:
        class_doc = docs.get(class_name, "")
        constructor_doc = docs.get(f"{class_name}_python", "")
        method_docs = {
            k: v for k, v in docs.items() if k not in (class_name, f"{class_name}_python")
        }
        return {
            "class_name": class_name,
            "class_doc": class_doc,
            "constructor_doc": constructor_doc,
            "constructor_params": _parse_numpydoc_params(constructor_doc),
            "method_docs": method_docs,
        }

    # Find all namespace declarations.
    # Namespaces without "::" are inner per-class namespaces (e.g. "namespace AJASourceOp {").
    # Namespaces with "::" are outer/qualified namespaces (e.g. "namespace holoscan::doc {").
    all_ns = re.findall(r"namespace\s+([\w:]+)\s*\{", content)
    inner_namespaces = [ns for ns in all_ns if "::" not in ns]

    if inner_namespaces:
        # Multi-class file: emit one record per inner namespace block so that
        # repeated PYDOC keys (e.g. "initialize") in different namespaces don't
        # overwrite each other.
        results = []
        for ns_name in inner_namespaces:
            # Prefer the closing-comment form "} // namespace NS" as a reliable
            # block terminator; fall back to the first closing brace otherwise.
            block_match = re.search(
                r"namespace\s+"
                + re.escape(ns_name)
                + r"\s*\{(.+?)\}\s*//\s*namespace\s+"
                + re.escape(ns_name),
                content,
                re.DOTALL,
            )
            if not block_match:
                block_match = re.search(
                    r"namespace\s+" + re.escape(ns_name) + r"\s*\{(.+?)\}",
                    content,
                    re.DOTALL,
                )
            if not block_match:
                continue
            docs = {
                m.group(1): m.group(2).strip() for m in pydoc_pattern.finditer(block_match.group(1))
            }
            if docs:
                results.append(_make_record(ns_name, docs))
        return results

    # Single fully-qualified namespace (e.g. "namespace holoscan::doc::ClassName {").
    # Use the last segment of the last namespace match as the class name and parse
    # all PYDOC entries from the whole file.
    if not all_ns:
        return []
    class_name = all_ns[-1].split("::")[-1]
    docs = {m.group(1): m.group(2).strip() for m in pydoc_pattern.finditer(content)}
    if not docs:
        return []
    return [_make_record(class_name, docs)]


def _parse_numpydoc_params(docstring: str) -> list[dict]:
    """Parse Parameters section from a numpydoc-style docstring.

    Returns a list of dicts with name, type, optional, default, description.
    """
    params = []
    if not docstring:
        return params

    # Find the Parameters section
    lines = docstring.split("\n")
    in_params = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        if stripped == "Parameters":
            in_params = True
            continue
        if stripped.startswith("---") and in_params:
            continue

        # Exit on next section header (Returns, Notes, etc.)
        if in_params and stripped and not line.startswith(" "):
            if re.match(r"^[A-Z]\w+$", stripped):
                break

        if not in_params:
            continue

        # Parameter line: "name : type" or "name : type, optional"
        param_match = re.match(r"^(\w+)\s*:\s*(.+)$", stripped)
        if param_match:
            if current_param:
                params.append(current_param)
            pname = param_match.group(1)
            ptype_str = param_match.group(2).strip()
            optional = "optional" in ptype_str
            ptype = re.sub(r",?\s*optional\s*$", "", ptype_str).strip()
            current_param = {
                "name": pname,
                "type": ptype,
                "optional": optional,
                "description": "",
            }
        elif current_param and stripped:
            # Continuation of description
            desc = current_param["description"]
            current_param["description"] = (desc + " " + stripped).strip()

    if current_param:
        params.append(current_param)

    return params


# ──────────────────────────────────────────────────────────────────────────────
# Markdown generation
# ──────────────────────────────────────────────────────────────────────────────


def _format_cpp_class_md(class_info: dict) -> str:
    """Format a single C++ class as Markdown API reference."""
    lines = []
    name = class_info["name"]
    bases = class_info.get("base_classes", [])
    desc = class_info.get("description", "")

    lines.append(f"#### `{name}`")
    lines.append("")
    if bases:
        bases_str = ", ".join(f"`{b}`" for b in bases)
        lines.append(f"**Inherits from:** {bases_str}")
        lines.append("")
    if desc:
        lines.append(desc)
        lines.append("")

    # Methods
    methods = class_info.get("methods", [])
    if methods:
        lines.append("##### Methods")
        lines.append("")
        lines.append("| Method | Description |")
        lines.append("|--------|-------------|")
        for m in methods:
            ret = m.get("return_type", "void")
            argsstring = m.get("argsstring", "()")
            # Doxygen leaves constructor <type> empty; omit ret to avoid a leading space.
            sig = f"`{m['name']}{argsstring}`" if not ret else f"`{ret} {m['name']}{argsstring}`"
            mdesc = m.get("description", "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {sig} | {mdesc} |")
        lines.append("")

    # Parameters (Holoscan Parameter<T> members)
    parameters = class_info.get("parameters", [])
    if parameters:
        lines.append("##### Parameters")
        lines.append("")
        lines.append("| Name | Type | Description |")
        lines.append("|------|------|-------------|")
        for p in parameters:
            pname = p["name"].rstrip("_")
            ptype = f"`{p['type']}`"
            pdesc = p.get("description", "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| `{pname}` | {ptype} | {pdesc} |")
        lines.append("")

    return "\n".join(lines)


def _format_python_class_md(class_info: dict) -> str:
    """Format a Python operator class (from AST parsing) as Markdown API reference."""
    lines = []
    name = class_info["name"]
    bases = class_info.get("base_classes", [])
    desc = class_info.get("description", "")

    lines.append(f"#### `{name}`")
    lines.append("")
    if bases:
        bases_str = ", ".join(f"`{b}`" for b in bases)
        lines.append(f"**Inherits from:** {bases_str}")
        lines.append("")
    if desc:
        lines.append(desc)
        lines.append("")

    # Constructor params (class-level annotations)
    parameters = class_info.get("parameters", [])
    if parameters:
        lines.append("##### Attributes")
        lines.append("")
        lines.append("| Name | Type | Default |")
        lines.append("|------|------|---------|")
        for p in parameters:
            ptype = f"`{p['type']}`" if p.get("type") else ""
            default = f"`{p['default']}`" if p.get("default") else ""
            lines.append(f"| `{p['name']}` | {ptype} | {default} |")
        lines.append("")

    # Methods
    methods = class_info.get("methods", [])
    if methods:
        lines.append("##### Methods")
        lines.append("")
        lines.append("| Method | Description |")
        lines.append("|--------|-------------|")
        for m in methods:
            params_str = ", ".join(p["name"] for p in m.get("params", []))
            sig = f"`{m['name']}({params_str})`"
            mdesc = m.get("description", "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {sig} | {mdesc} |")
        lines.append("")

    return "\n".join(lines)


def _format_pydoc_class_md(pydoc_info: dict) -> str:
    """Format a pybind11 operator class (from pydoc.hpp parsing) as Markdown API reference."""
    lines = []
    class_name = pydoc_info["class_name"]
    class_doc = pydoc_info.get("class_doc", "")
    constructor_params = pydoc_info.get("constructor_params", [])
    method_docs = pydoc_info.get("method_docs", {})

    lines.append(f"#### `{class_name}`")
    lines.append("")
    if class_doc:
        lines.append(class_doc)
        lines.append("")

    if constructor_params:
        lines.append("##### Constructor Parameters")
        lines.append("")
        lines.append("| Parameter | Type | Required | Description |")
        lines.append("|-----------|------|----------|-------------|")
        for p in constructor_params:
            ptype = f"`{p['type']}`" if p.get("type") else ""
            required = "Optional" if p.get("optional") else "Required"
            pdesc = p.get("description", "").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| `{p['name']}` | {ptype} | {required} | {pdesc} |")
        lines.append("")

    if method_docs:
        lines.append("##### Methods")
        lines.append("")
        for method_name, doc in method_docs.items():
            first_line = doc.split("\n")[0] if doc else ""
            lines.append(f"- **`{method_name}`**: {first_line}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main: build operator -> API reference mapping
# ──────────────────────────────────────────────────────────────────────────────


def _find_operator_dir_for_header(header_path: str, git_repo_path: Path) -> str | None:
    """Map a C++ header file path back to its most specific operator directory.

    Searches from the header's parent directory upward, finding the deepest
    directory that contains a metadata.json. This ensures headers in nested
    operator directories (e.g. video_streaming/video_streaming_client/) are
    mapped to the correct sub-operator, not a parent.

    Returns the operator directory path relative to git_repo_path (e.g. "operators/gamma_correction")
    or None if it can't be determined.
    """
    # header_path is relative to the Doxyfile working directory (doc/website/)
    # It typically looks like "../../operators/xyz/abc.hpp"
    header = Path(header_path)
    try:
        resolved = (git_repo_path / "doc" / "website" / header).resolve()
        rel = resolved.relative_to(git_repo_path)
    except (ValueError, OSError):
        # Try treating it as already relative to repo root
        rel = header

    parts = rel.parts
    if len(parts) < 2 or parts[0] != "operators":
        return None

    # Walk from the header's parent directory upward toward "operators/",
    # returning the deepest directory that has a metadata.json.
    for i in range(len(parts) - 1, 0, -1):
        candidate = git_repo_path / Path(*parts[:i])
        # Skip cpp/python language directories themselves
        if parts[i - 1] in ("cpp", "python", "include"):
            continue
        if (candidate / "metadata.json").exists():
            return str(Path(*parts[:i]))

    # Fallback: strip trailing language/include directories and return the deepest
    # valid operator path, preserving full nesting (e.g. operators/group/operator).
    fallback_parts = list(parts[:-1])  # drop the filename
    while fallback_parts and fallback_parts[-1] in ("cpp", "python", "include"):
        fallback_parts.pop()
    if len(fallback_parts) >= 2 and fallback_parts[0] == "operators":
        return str(Path(*fallback_parts))
    return None


def build_api_reference_map(git_repo_path: Path) -> dict:
    """Build a mapping of operator directory paths to API reference Markdown content.

    Args:
        git_repo_path: Absolute path to the repository root.

    Returns:
        Dict mapping operator directory paths (e.g. "operators/gamma_correction")
        to a string of Markdown API reference content.
    """
    website_dir = git_repo_path / "doc" / "website"
    operators_dir = git_repo_path / "operators"

    # ── C++ API via Doxygen ──
    header_to_classes = run_doxygen_and_parse(website_dir)

    # Map: operator_dir -> list of C++ class info
    op_cpp_classes: dict[str, list[dict]] = {}
    for header_path, classes in header_to_classes.items():
        op_dir = _find_operator_dir_for_header(header_path, git_repo_path)
        if op_dir:
            op_cpp_classes.setdefault(op_dir, []).extend(classes)

    # ── Python API ──
    # Strategy 1: Pure Python operators (parse .py files)
    op_python_classes: dict[str, list[dict]] = {}

    # Strategy 2: pybind11 operators (parse *_pydoc.hpp files)
    op_pydoc_classes: dict[str, list[dict]] = {}

    for metadata_path in operators_dir.rglob("metadata.json"):
        metadata_dir = metadata_path.parent

        # Look for pure Python sources.
        # When metadata_dir is itself a "python" directory, avoid building "python/python".
        python_dir = metadata_dir if metadata_dir.name == "python" else metadata_dir / "python"
        if python_dir.is_dir():
            # Check for pydoc.hpp files first (pybind11).
            # Collect both "*_pydoc.hpp" and bare "pydoc.hpp" naming conventions.
            pydoc_files: list[Path] = []
            for pattern in ("*_pydoc.hpp", "pydoc.hpp"):
                pydoc_files.extend(python_dir.glob(pattern))
            pydoc_files = list(dict.fromkeys(pydoc_files))  # deduplicate, preserve order
            if pydoc_files:
                for pydoc_file in pydoc_files:
                    infos = _parse_pydoc_hpp(pydoc_file)
                    if infos:
                        # Map to the parent operator directory (not the python subdir)
                        parent_op_dir = str(metadata_dir.relative_to(git_repo_path))
                        if metadata_dir.name in ("cpp", "python"):
                            parent_op_dir = str(metadata_dir.parent.relative_to(git_repo_path))
                        op_pydoc_classes.setdefault(parent_op_dir, []).extend(infos)
            else:
                # Pure Python: parse .py files
                for py_file in python_dir.glob("*.py"):
                    if py_file.name.startswith("_") or py_file.name == "setup.py":
                        continue
                    classes = _parse_python_source(py_file)
                    if classes:
                        parent_op_dir = str(metadata_dir.relative_to(git_repo_path))
                        if metadata_dir.name in ("cpp", "python"):
                            parent_op_dir = str(metadata_dir.parent.relative_to(git_repo_path))
                        op_python_classes.setdefault(parent_op_dir, []).extend(classes)

        # Also check for .py files directly in the operator dir (not in python/ subdir)
        for py_file in metadata_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "setup.py":
                continue
            classes = _parse_python_source(py_file)
            if classes:
                parent_op_dir = str(metadata_dir.relative_to(git_repo_path))
                if metadata_dir.name in ("cpp", "python"):
                    parent_op_dir = str(metadata_dir.parent.relative_to(git_repo_path))
                op_python_classes.setdefault(parent_op_dir, []).extend(classes)

    # ── Merge into final Markdown ──
    api_ref_map: dict[str, str] = {}

    all_op_dirs = (
        set(op_cpp_classes.keys()) | set(op_python_classes.keys()) | set(op_pydoc_classes.keys())
    )

    for op_dir in sorted(all_op_dirs):
        cpp_classes = op_cpp_classes.get(op_dir, [])
        py_classes = op_python_classes.get(op_dir, [])
        pydoc_classes = op_pydoc_classes.get(op_dir, [])

        has_cpp = bool(cpp_classes)
        has_python = bool(py_classes) or bool(pydoc_classes)

        if not has_cpp and not has_python:
            continue

        sections = []
        sections.append("## API Reference\n")

        if has_cpp and has_python:
            # Use tabs for both languages
            sections.append('=== "C++"')
            sections.append("")
            for cls in cpp_classes:
                indented = _indent(_format_cpp_class_md(cls), 4)
                sections.append(indented)

            sections.append('=== "Python"')
            sections.append("")
            for cls in py_classes:
                indented = _indent(_format_python_class_md(cls), 4)
                sections.append(indented)
            for cls in pydoc_classes:
                indented = _indent(_format_pydoc_class_md(cls), 4)
                sections.append(indented)

        elif has_cpp:
            sections.append("### C++\n")
            for cls in cpp_classes:
                sections.append(_format_cpp_class_md(cls))

        else:
            sections.append("### Python\n")
            for cls in py_classes:
                sections.append(_format_python_class_md(cls))
            for cls in pydoc_classes:
                sections.append(_format_pydoc_class_md(cls))

        api_ref_map[op_dir] = "\n".join(sections)

    logger.info(f"Generated API references for {len(api_ref_map)} operators")
    return api_ref_map


def _indent(text: str, spaces: int) -> str:
    """Indent every line of text by the given number of spaces.

    Blank lines also receive the prefix so pymdownx.tabbed content blocks
    remain consistently indented (an unindented blank line would terminate
    the tab block early).
    """
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


# Module-level cache so generate_pages.py only calls this once per repo path
_cached_api_ref_map: dict[Path, dict] = {}


def get_api_reference_map(git_repo_path: Path) -> dict:
    """Get (cached) API reference map. Safe to call multiple times."""
    if git_repo_path not in _cached_api_ref_map:
        _cached_api_ref_map[git_repo_path] = build_api_reference_map(git_repo_path)
    return _cached_api_ref_map[git_repo_path]


def get_api_reference_for_operator(op_dir: str, git_repo_path: Path) -> str:
    """Get the API reference Markdown for a specific operator directory.

    Args:
        op_dir: Operator directory relative to repo root (e.g. "operators/gamma_correction").
        git_repo_path: Absolute path to the repository root.

    Returns:
        Markdown string of the API reference section, or empty string if none.
    """
    ref_map = get_api_reference_map(git_repo_path)

    # Try exact match first
    if op_dir in ref_map:
        return ref_map[op_dir]

    # Try parent directory (for cpp/python subdirs)
    parent = str(Path(op_dir).parent)
    if parent in ref_map:
        return ref_map[parent]

    return ""
