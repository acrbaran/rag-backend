import logging

from common import config_utils


def test_show_configs_logs_section_names_without_values(monkeypatch, caplog):
    marker = "do-not-log-nested-secret"
    monkeypatch.setattr(
        config_utils,
        "CONFIGS",
        {
            "mysql": {"host": "mysql", "config": {"password": marker}},
            "redis": {"host": "redis"},
        },
    )

    with caplog.at_level(logging.INFO):
        config_utils.show_configs()

    assert marker not in caplog.text
    assert "host" not in caplog.text
    assert "mysql" in caplog.text
    assert "redis" in caplog.text
    assert "section_count=2" in caplog.text
