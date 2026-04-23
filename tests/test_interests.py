from pathlib import Path

from recommender.interests import load


def test_load_reads_memory_md_verbatim(tmp_path: Path):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("# interests\nfoo\nbar\n")
    primary, secondary = load(memory_md=memory_md, claude_projects_root=tmp_path / "nope")
    assert primary == "# interests\nfoo\nbar\n"
    assert secondary == ""


def test_load_scans_claude_code_memory_dirs(tmp_path: Path):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("primary")

    projects = tmp_path / "projects"
    proj_a = projects / "-home-user-src-repo-a" / "memory"
    proj_a.mkdir(parents=True)
    (proj_a / "MEMORY.md").write_text("a-memory")
    (proj_a / "project_foo.md").write_text("proj-foo-note")

    proj_b = projects / "-home-user-src-repo-b" / "memory"
    proj_b.mkdir(parents=True)
    (proj_b / "user_role.md").write_text("user-role-b")

    (projects / "not-a-project-dir").mkdir()   # no memory/, should be skipped

    primary, secondary = load(memory_md=memory_md, claude_projects_root=projects)
    assert primary == "primary"
    assert "repo-a" in secondary
    assert "a-memory" in secondary
    assert "proj-foo-note" in secondary
    assert "repo-b" in secondary
    assert "user-role-b" in secondary
