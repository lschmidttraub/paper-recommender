from __future__ import annotations

from pathlib import Path

_RELEVANT_PREFIXES = ("MEMORY.md", "project_", "user_", "feedback_", "reference_")


def load(memory_md: Path, claude_projects_root: Path) -> tuple[str, str]:
    primary = memory_md.read_text() if memory_md.exists() else ""
    secondary = _scan_claude_memory(claude_projects_root)
    return primary, secondary


def _scan_claude_memory(root: Path) -> str:
    if not root.exists():
        return ""
    blocks: list[str] = []
    for project_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        memory_dir = project_dir / "memory"
        if not memory_dir.is_dir():
            continue
        project_name = project_dir.name
        files = sorted(
            f for f in memory_dir.iterdir()
            if f.is_file() and f.suffix == ".md"
            and (f.name == "MEMORY.md" or any(f.name.startswith(p) for p in _RELEVANT_PREFIXES))
        )
        if not files:
            continue
        body = "\n\n".join(f"## {f.name}\n{f.read_text().strip()}" for f in files)
        blocks.append(f"# Project: {project_name}\n\n{body}")
    return "\n\n---\n\n".join(blocks)
