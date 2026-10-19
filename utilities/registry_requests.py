#!/usr/bin/env python3
"""Turn registration issue forms into registry pull requests."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry

try:
    from .validate_registry import KIND_DIRECTORIES, SCHEMA_PATH, read_json, source_identity, validate_registry
except ImportError:
    from validate_registry import KIND_DIRECTORIES, SCHEMA_PATH, read_json, source_identity, validate_registry


# Form input ids and the labels GitHub renders as issue headings.
FIELDS = {"url": "Source URL", "ref": "Git revision", "path": "Repository path", "issues-url": "Issue tracker URL"}
LABELS = ("Kind", "Name", *FIELDS.values(), "Remove", "Reason")
KINDS = {**{kind: kind for kind in KIND_DIRECTORIES}, **{folder: kind for kind, folder in KIND_DIRECTORIES.items()}}
GUIDANCE = {
    "name": "Name must use lowercase letters and numbers, with single hyphens or underscores between words.",
    "issues_url": "Issue tracker URL must be an HTTPS URL without credentials.",
    "source/url": "Source URL must be an HTTPS URL without credentials; a Git URL cannot include ? or #.",
    "source/ref": "Git revision must be a branch, tag, or commit without spaces that does not start with -.",
    "source/path": "Repository path must be a relative directory such as applications/my-app, or . for the root.",
}
FEEDBACK = "<!-- holohub-registry-feedback -->"
TITLE_MARK = "🗂️"
BOT = "github-actions[bot]"


class RequestError(ValueError):
    """A problem the requester can fix by editing the issue."""


def parse_issue(body):
    """Read the sections GitHub renders for the form; Reason, the last one, is free text."""
    if not isinstance(body, str):
        raise RequestError("Use the registration form and keep its headings.")
    if len(body) > 65536:
        raise RequestError("Shorten the issue; its body is limited to 65,536 characters.")
    body = body.replace("\r\n", "\n").strip()
    headings = list(re.finditer(r"^### (.+)\n", body, re.MULTILINE))
    if not headings or headings[0].start():
        raise RequestError("Use the registration form and keep its headings.")
    sections = {}
    for heading, following in zip(headings, [*headings[1:], None]):
        label = heading[1].strip()
        if label in sections:
            raise RequestError(f"The {label} heading appears more than once.")
        end = following.start() if following and label != "Reason" else len(body)
        value = body[heading.end():end].strip()
        sections[label] = "" if value == "_No response_" else value
        if label == "Reason":
            break
    if sections.keys() != set(LABELS):
        missing = ", ".join(label for label in LABELS if label not in sections) or "none"
        unexpected = ", ".join(label for label in sections if label not in LABELS) or "none"
        raise RequestError(f"Keep the form's headings. Missing: {missing}; unexpected: {unexpected}.")
    for label in LABELS[:-2]:
        if "\n" in sections[label]:
            raise RequestError(f"{label} must be a single line.")
    if not sections["Reason"]:
        raise RequestError("Give a reason for this request.")
    kind = KINDS.get(sections["Kind"].lower())
    if kind is None:
        raise RequestError(f"Kind must be one of: {', '.join(KIND_DIRECTORIES)}.")
    if not sections["Name"]:
        raise RequestError("Name is required.")
    return {
        "kind": kind,
        "name": sections["Name"].lower(),
        "fields": {field: sections[label] for field, label in FIELDS.items()},
        "remove": bool(re.search(r"^- \[[xX]\] ", sections["Remove"], re.MULTILINE)),
    }


def flatten(record):
    """Map a record onto the form's source fields."""
    source = record["source"]
    return {"url": source["url"], "ref": source.get("ref"), "path": source.get("path"), "issues-url": record.get("issues_url")}


def build(kind, name, fields):
    """Create a record in canonical key order; a Git revision makes it a Git source."""
    record = {"name": name, "kind": kind}
    if fields["issues-url"]:
        record["issues_url"] = fields["issues-url"]
    record["source"] = {"type": "git" if fields["ref"] else "url", "url": fields["url"]}
    for key in ("ref", "path"):
        if fields[key]:
            record["source"][key] = fields[key]
    return record


def looks_like_git_repository(url):
    """An HTTPS clone URL: it ends in .git, or it is a GitHub owner/repository page."""
    parts = urlsplit(url)
    return parts.scheme == "https" and (parts.path.endswith(".git") or (parts.hostname == "github.com" and parts.path.strip("/").count("/") == 1))


def default_branch(url):
    """Ask the repository which branch HEAD points to, without credentials or prompts."""
    command = ["git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "ls-remote", "--symref", "--", url, "HEAD"]
    try:
        result = subprocess.run(command, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, text=True, capture_output=True, timeout=60, check=True)
        found = re.search(r"^ref: refs/heads/(\S+)\tHEAD$", result.stdout, re.MULTILINE)
    except (OSError, subprocess.SubprocessError):
        found = None
    if found is None:
        raise RequestError(f"Could not find the default branch of {url}. Check that the repository is public, or enter a Git revision.")
    return found[1]


def hosted_project(url):
    """The host and project path of a github.com or gitlab.com repository URL, or None for anything else."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ("github.com", "gitlab.com") or parts.port or parts.query or parts.fragment:
        return None
    segments = parts.path.strip("/").removesuffix(".git").split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", segment) or segment in (".", "..", "-") for segment in segments):
        return None  # "-" marks GitLab pages such as /-/tree/main, not a project.
    if len(segments) != 2 and (parts.hostname == "github.com" or len(segments) < 2):
        return None
    return parts.hostname, "/".join(segments)


def fetch_json(url, token=None):
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "holohub-registry"})
    if token:
        # An unredirected header is never forwarded if the API redirects elsewhere.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def issue_tracker(url):
    """The public issue tracker of a GitHub or GitLab repository, or None when it is off or cannot be checked."""
    project = hosted_project(url)
    if project is None:
        return None
    host, path = project
    try:
        if host == "github.com":
            data = fetch_json(f"https://api.github.com/repos/{path}", os.environ.get("GH_TOKEN"))
            page, open_ = data["html_url"], data.get("has_issues") and not data.get("archived") and not data.get("private")
            tracker = f"{page}/issues"
        else:
            # Anonymous project responses omit the issue settings, so ask for the public issue list itself;
            # GitLab refuses it when issues are disabled or limited to members.
            project_api = f"https://gitlab.com/api/v4/projects/{quote(path, safe='')}"
            data = fetch_json(project_api)
            page, open_ = data["web_url"], not data.get("archived") and isinstance(fetch_json(f"{project_api}/issues?per_page=1"), list)
            tracker = f"{page}/-/issues"
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return tracker if open_ and isinstance(page, str) and page.startswith(f"https://{host}/") else None


def complete(fields, resolve, notes):
    """A Git repository URL without a revision takes the repository's default branch."""
    if fields["url"] and not fields["ref"] and looks_like_git_repository(fields["url"]):
        fields = {**fields, "ref": resolve(fields["url"])}
        notes.append(f"Git revision: {fields['ref']} (the repository's default branch)")
    return fields


def new_record(kind, name, fields, resolve=default_branch, notes=None):
    fields = {field: None if value.lower() == "none" else value or None for field, value in fields.items()}
    if not fields["url"]:
        raise RequestError(f"{kind}/{name} is not registered yet. Check its kind and name, or add a Source URL to register it.")
    return build(kind, name, complete(fields, resolve, [] if notes is None else notes))


def updated_record(current, fields, resolve=default_branch, notes=None):
    """Blank fields keep their registered values, and none removes an optional value."""
    merged = flatten(current)
    for field, value in fields.items():
        if value.lower() == "none":
            if field == "url":
                raise RequestError("Source URL cannot be removed; enter the new URL instead.")
            merged[field] = None
        elif value:
            merged[field] = value
    if fields["ref"].lower() != "none":
        merged = complete(merged, resolve, [] if notes is None else notes)
    if not merged["ref"] and not fields["path"]:
        merged["path"] = None  # Hosted content has no repository path.
    return build(current["kind"], current["name"], merged)


def check(root, record, others):
    """Validate a proposed record against the schema and the other registrations."""
    source = record["source"]
    if source["type"] == "url" and "path" in source:
        raise RequestError("Repository path applies only to Git sources; add a Git revision or leave the path blank.")
    validator = Draft202012Validator(read_json(root / SCHEMA_PATH), format_checker=FormatChecker(), registry=Registry())
    problems = []
    for error in validator.iter_errors(record):
        pointer = "/".join(map(str, error.absolute_path))
        problems.append(GUIDANCE.get(pointer, f"{pointer or 'record'}: {error.message}"))
    if problems:
        raise RequestError("\n".join(dict.fromkeys(problems)))
    if source["type"] == "url" and looks_like_git_repository(source["url"]):
        raise RequestError("Source URL looks like a Git repository, which needs a Git revision; "
                           "leave the field blank to use its default branch.")
    if source["type"] == "git" and urlsplit(source["url"]).path.endswith(("/metadata.json", "/SKILL.md")):
        raise RequestError("Source URL links to a hosted metadata.json or SKILL.md, which takes no Git revision. "
                           "Leave Git revision blank for a new source, or enter none to switch a registered one.")
    identity = source_identity(record)
    for other in others:
        if source_identity(other) == identity:
            raise RequestError(f"This source is already registered as {other['kind']}/{other['name']}; update that registration instead.")


def summarize(current, record):
    """List the form fields a request sets or changes."""
    if record is None:
        return []
    before = flatten(current) if current else {}
    lines = []
    for field, value in flatten(record).items():
        old = before.get(field)
        if value != old:
            lines.append(f"{FIELDS[field]}: {value}" if current is None else f"{FIELDS[field]}: {old or '(none)'} → {value or '(none)'}")
    return lines


def load_records(root):
    records, errors = validate_registry(root)
    if errors:
        raise ValueError("Registry validation failed:\n" + "\n".join(errors))
    return records


def prepare(root, body, resolve=default_branch, tracker=issue_tracker):
    """Apply a valid request to the working tree; an invalid one changes nothing.

    Blank fields that the source can answer are filled in, and each is noted for the requester.
    """
    request = parse_issue(body)
    kind, name = request["kind"], request["name"]
    key = f"{kind}/{name}"
    records = load_records(root)
    current = next((record for record in records if record["kind"] == kind and record["name"] == name), None)
    others = [record for record in records if record is not current]
    notes = []
    if request["remove"]:
        if current is None:
            raise RequestError(f"{key} is not registered, so there is nothing to remove.")
        action, record = "Remove", None
    elif current is None:
        action, record = "Register", new_record(kind, name, request["fields"], resolve, notes)
    else:
        action, record = "Update", updated_record(current, request["fields"], resolve, notes)
        if record == current:
            return {"status": "noop", "registration": key}
    if record is not None:
        check(root, record, others)
        # Only a valid request reaches the network, and only for a new or moved Git source left without a tracker.
        moved = current is None or record["source"]["url"] != current["source"]["url"]
        if moved and record["source"]["type"] == "git" and "issues_url" not in record and not request["fields"]["issues-url"]:
            if found := tracker(record["source"]["url"]):
                record = build(kind, name, {**flatten(record), "issues-url": found})
                check(root, record, others)
                notes.append(f"Issue tracker URL: {found} (the repository's issue tracker)")
    # The name is validated or already registered, so it always maps inside its kind's directory.
    path = root / KIND_DIRECTORIES[kind] / f"{name}.json"
    if record is None:
        path.unlink()
        if not any(path.parent.iterdir()):
            # Git does not keep empty directories, and the registry requires each one.
            (path.parent / ".gitkeep").touch()
    else:
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "status": "ready", "registration": key, "title": f"{action} {key}",
        "directory": KIND_DIRECTORIES[kind], "changes": summarize(current, record), "notes": notes,
    }


def issue_title(root, body):
    """Name the issue after its request, such as "🗂️ [NEW] application: camera"; None when the form cannot be read."""
    try:
        request = parse_issue(body)
    except RequestError:
        return None
    kind, name = request["kind"], request["name"]
    registered = any(record["kind"] == kind and record["name"] == name for record in load_records(root))
    action = "REMOVE" if request["remove"] else "UPDATE" if registered else "NEW"
    # GitHub limits titles to 256 characters; a name that long fails validation anyway.
    return f"{TITLE_MARK} [{action}] {kind}: {name[:200]}"


def label_changes(body, labels):
    """Label the issue with its kind, such as "module", and drop the label of a kind it no longer requests."""
    try:
        kind = parse_issue(body)["kind"]
    except RequestError:
        return None
    changes = {"add": [kind] if kind not in labels else [], "remove": [label for label in labels if label in KIND_DIRECTORIES and label != kind]}
    return changes if changes["add"] or changes["remove"] else None


def commit_message(title, issue):
    """The proposal's commit message, crediting the requester as a co-author.

    The bot stays the author because the DCO check exempts bot-authored commits. The requester's login
    and numeric id come from GitHub, not the form, and anything other than a regular user is left out.
    """
    user = issue.get("user") or {}
    login, number = user.get("login") or "", user.get("id")
    if user.get("type") != "User" or type(number) is not int or not re.fullmatch(r"[A-Za-z0-9](?:-?[A-Za-z0-9]){0,38}", login):
        return title
    return f"{title}\n\nCo-authored-by: {login} <{number}+{login}@users.noreply.github.com>"


def pull_request_body(result, number):
    changes = "".join(f"    {line}\n" for line in result["changes"])
    return (f"{changes}\n" if changes else "") + f"Closes #{number}.\n"


def github_api(endpoint, method="GET", data=None):
    """Call the GitHub REST API with the runner's authenticated GitHub CLI."""
    command = ["gh", "api", "--method", method, endpoint]
    if data is not None:
        command.extend(["--input", "-"])
    result = subprocess.run(
        command, input=None if data is None else json.dumps(data),
        text=True, capture_output=True, timeout=60, check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def report(repository, number, result, branch, pull_url="", run_url="", failed=False, api=github_api):
    """Keep one feedback comment on the issue, and withdraw a proposal that no longer matches it."""
    status = result.get("status", "error")
    if result.get("issue_title"):
        # The workflow token's edits start no new run, so this cannot loop.
        api(f"repos/{repository}/issues/{number}", "PATCH", {"title": result["issue_title"]})
    labels = result.get("issue_labels") or {}
    if labels.get("add"):
        api(f"repos/{repository}/issues/{number}/labels", "POST", {"labels": labels["add"]})
    for label in labels.get("remove", []):
        api(f"repos/{repository}/issues/{number}/labels/{quote(label, safe='')}", "DELETE")
    failed = failed or (status == "ready" and not pull_url)
    withdrawn = []
    if not failed and status in ("noop", "error", "closed"):
        query = urlencode({"state": "open", "head": f"{repository.split('/')[0]}:{branch}"})
        withdrawn = [pull for pull in api(f"repos/{repository}/pulls?{query}") if pull["user"]["login"] == BOT]
        for pull in withdrawn:
            api(f"repos/{repository}/pulls/{pull['number']}", "PATCH", {"state": "closed"})
    if failed:
        parts = [f"The registry automation did not finish; a maintainer can check [the workflow run]({run_url}). "
                 "Edit or reopen this issue to retry."]
        if pull_url:
            parts.append(f"Pull request: {pull_url}")
    elif status == "ready":
        parts = [f"**{result['title']}** is ready for review in {pull_url}."]
        if result["changes"]:
            # Indented code keeps requester-supplied values from rendering as Markdown or mentions.
            parts.append("\n".join(f"    {line}" for line in result["changes"]))
        if result.get("notes"):
            parts.append("These fields were blank, so they were filled in from the source. "
                         "To use different values, edit this issue:")
            parts.append("\n".join(f"    {line}" for line in result["notes"]))
        parts.append("The change takes effect when a maintainer merges the pull request. "
                     "To revise the request, edit this issue.")
    elif status == "noop":
        parts = [f"**{result['registration']}** already matches this request, so nothing needs to change. "
                 "Blank fields keep their current values; enter `none` to remove an optional value."]
    elif status == "closed":
        if not withdrawn:
            return  # Merged or never proposed: keep the latest feedback.
        parts = ["This request was closed, so its pull request was closed too. Reopen the issue to try again."]
    else:
        parts = [
            "This request needs a change before it can be proposed. Edit the issue to fix the following:",
            "\n".join(f"    {line}" for line in result.get("message", "Invalid request.")[:6000].splitlines()),
        ]
    if withdrawn and status != "closed":
        parts.append("The previous pull request was closed.")
    body = FEEDBACK + "\n" + "\n\n".join(parts)
    comment = None
    page = 1
    while comment is None:
        comments = api(f"repos/{repository}/issues/{number}/comments?per_page=100&page={page}")
        comment = next((item for item in comments if item["user"]["login"] == BOT and (item.get("body") or "").startswith(FEEDBACK)), None)
        if len(comments) < 100:
            break
        page += 1
    if comment is None:
        api(f"repos/{repository}/issues/{number}/comments", "POST", {"body": body})
    elif comment["body"] != body:
        api(f"repos/{repository}/issues/comments/{comment['id']}", "PATCH", {"body": body})
    if withdrawn:
        api(f"repos/{repository}/git/refs/heads/{branch}", "DELETE")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registry = argparse.ArgumentParser(add_help=False)
    registry.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="registry root (default: this repository)")
    request = commands.add_parser("prepare", parents=[registry], help="apply a registration issue to the working tree")
    request.add_argument("--issue", type=Path, required=True, help="issue JSON from the GitHub REST API")
    request.add_argument("--result", type=Path, required=True, help="where to write the outcome as JSON")
    request.add_argument("--pull-request-body", type=Path, required=True, help="where to write the pull request description")
    feedback = commands.add_parser("report", help="comment on the issue with the outcome of prepare")
    feedback.add_argument("--repository", required=True, help="owner/name")
    feedback.add_argument("--issue-number", type=int, required=True)
    feedback.add_argument("--branch", required=True, help="the request's pull request branch")
    feedback.add_argument("--result", type=Path, required=True, help="outcome JSON from prepare")
    feedback.add_argument("--pull-request", default="", help="URL of the request's pull request")
    feedback.add_argument("--run-url", default="", help="URL of the workflow run")
    feedback.add_argument("--job-status", default="success", help="the job status so far")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        issue = read_json(args.issue)
        if issue["state"] == "closed":
            result = {"status": "closed"}
        else:
            # Name the issue from the registry before prepare applies the request to it.
            title = issue_title(args.root, issue["body"])
            try:
                result = prepare(args.root, issue["body"])
            except RequestError as error:
                result = {"status": "error", "message": str(error)}
            if result["status"] == "ready":
                args.pull_request_body.write_text(pull_request_body(result, int(issue["number"])), encoding="utf-8")
            if title and title != issue.get("title"):
                result["issue_title"] = title
            if labels := label_changes(issue["body"], [label["name"] for label in issue.get("labels") or []]):
                result["issue_labels"] = labels
        args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a", encoding="utf-8") as stream:
                for key in ("status", "title", "directory"):
                    if key in result:
                        print(f"{key}={result[key]}", file=stream)
                if result["status"] == "ready":
                    # Multiline output syntax; the message holds only the title and a validated trailer.
                    print(f"commit_message<<END_OF_MESSAGE\n{commit_message(result['title'], issue)}\nEND_OF_MESSAGE", file=stream)
        return 0

    try:
        result = json.loads(args.result.read_text(encoding="utf-8")) if args.result.is_file() else {}
        report(args.repository, args.issue_number, result, args.branch, args.pull_request, args.run_url, args.job_status != "success")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Could not report the registry request: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
