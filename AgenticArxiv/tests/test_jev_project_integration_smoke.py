"""End-to-end contract proof for switching policy routing to Jev routing."""

from pathlib import Path

from scripts.jev_project_integration_smoke import run


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_policy_and_jev_contract_paths_are_end_to_end_switch_compatible():
    result = run(
        REPO_ROOT
        / "AgenticArxiv"
        / "tests"
        / "fixtures"
        / "jev_integration_snapshot.json",
        REPO_ROOT / "artifacts" / "jev_route_expanded_all.json",
    )

    assert all(result["checks"].values())
    assert result["policy"]["strict_success"] is True
    assert result["jev"]["strict_success"] is True
    assert result["policy"]["tool_sequence"] == result["jev"]["tool_sequence"]
    assert result["policy"]["routing"]["mode"] == "policy"
    assert result["jev"]["routing"]["mode"] == "jev"
    assert result["benchmark_summary"]["jev"]["router_used"] == 2
    assert result["live_routing_pilot"]["tasks"] == 81
    assert result["live_routing_pilot"]["correct"] == 70
