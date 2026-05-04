from pathlib import Path

from recommender.interests import load
from recommender.sources.zotero import ZoteroItem


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


def test_load_appends_zotero_when_credentials_set(tmp_path: Path, mocker):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("# interests\n- X\n")

    fake_items = [
        ZoteroItem(
            key="K", title="Some Saved Paper", creators=("Alice",),
            year="2024", tags=("ml",), item_type="journalArticle",
            date_added="2024-01-01T00:00:00Z",
        ),
    ]
    mocker.patch("recommender.interests.fetch_items", return_value=fake_items)

    primary, _ = load(
        memory_md=memory_md,
        claude_projects_root=tmp_path / "no",
        zotero_api_key="k",
        zotero_user_id="123",
    )
    assert "# interests" in primary
    assert "## Your Zotero library" in primary
    assert "Some Saved Paper" in primary


def test_load_skips_zotero_when_credentials_missing(tmp_path: Path, mocker):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("# interests")
    fetch_mock = mocker.patch("recommender.interests.fetch_items")
    primary, _ = load(memory_md=memory_md, claude_projects_root=tmp_path / "no")
    assert "Zotero" not in primary
    fetch_mock.assert_not_called()


def test_load_swallows_zotero_fetch_errors(tmp_path: Path, mocker):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("# interests")
    mocker.patch("recommender.interests.fetch_items", side_effect=RuntimeError("boom"))
    primary, _ = load(
        memory_md=memory_md,
        claude_projects_root=tmp_path / "no",
        zotero_api_key="k", zotero_user_id="123",
    )
    assert "# interests" in primary
    assert "Zotero" not in primary  # fetch failure → no Zotero section
