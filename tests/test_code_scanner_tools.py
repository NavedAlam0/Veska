"""Tests for built-in code scanner tools."""

import asyncio

from veska.core.agent import Agent
from veska.tools.registry import ToolRegistry


def test_registry_loads_code_scanner_tools():
    registry = ToolRegistry()
    registry.register("code_scanner")

    assert set(registry.get_names()) == {
        "scan_code",
        "scan_file",
        "scan_directory",
    }


def test_scan_code_reports_security_warnings():
    registry = ToolRegistry()
    registry.register("code_scanner")

    result = asyncio.run(
        registry.get("scan_code").execute(
            code='password = "secret123"\neval(user_input)',
            language="python",
        )
    )

    assert result.success
    assert not result.output["clean"]
    assert result.output["warning_count"] == 2


def test_scan_file_reports_security_warnings(tmp_path):
    file_path = tmp_path / "app.py"
    file_path.write_text('api_key = "abcdefghijklmnopqrstuvwxyz"\n')
    registry = ToolRegistry()
    registry.register("code_scanner")

    result = asyncio.run(
        registry.get("scan_file").execute(
            path=str(file_path),
            language="python",
        )
    )

    assert result.success
    assert not result.output["clean"]
    assert result.output["warnings"][0]["line"] == 1


def test_scan_directory_scans_supported_files(tmp_path):
    (tmp_path / "safe.txt").write_text("password = 'ignored because extension'")
    (tmp_path / "server.py").write_text("DEBUG = True\n")
    registry = ToolRegistry()
    registry.register("code_scanner")

    result = asyncio.run(
        registry.get("scan_directory").execute(directory=str(tmp_path))
    )

    assert result.success
    assert result.output["files_scanned"] == 1
    assert result.output["warning_count"] == 1
    assert result.output["warnings"][0]["file"].endswith("server.py")


def test_agent_loads_code_scanner_by_prebuilt_name():
    agent = Agent(name="security", provider=object(), tools=["code_scanner"])

    assert {tool.name for tool in agent.tools} == {
        "scan_code",
        "scan_file",
        "scan_directory",
    }
