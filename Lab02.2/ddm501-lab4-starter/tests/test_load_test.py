"""Tests for load-test safety checks and URL configuration."""

import sys

from scripts import load_test


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


def test_health_requires_loaded_healthy_model(monkeypatch):
    responses = [
        FakeResponse(200, {"status": "healthy", "model_loaded": True}),
        FakeResponse(200, {"status": "unhealthy", "model_loaded": False}),
        FakeResponse(503, {"status": "unhealthy", "model_loaded": False}),
    ]
    monkeypatch.setattr(
        load_test.requests, "get", lambda *args, **kwargs: responses.pop(0)
    )

    assert load_test.check_health() is True
    assert load_test.check_health() is False
    assert load_test.check_health() is False


def test_health_rejects_invalid_json_and_connection_errors(monkeypatch):
    class InvalidJsonResponse(FakeResponse):
        def json(self):
            raise ValueError("invalid json")

    monkeypatch.setattr(
        load_test.requests,
        "get",
        lambda *args, **kwargs: InvalidJsonResponse(),
    )
    assert load_test.check_health() is False

    monkeypatch.setattr(
        load_test.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConnectionError("offline")
        ),
    )
    assert load_test.check_health() is False


def test_cli_url_overrides_default(monkeypatch):
    captured = {}

    def fake_run(duration, workers, batch, allow_unhealthy):
        captured.update(
            duration=duration,
            workers=workers,
            batch=batch,
            allow_unhealthy=allow_unhealthy,
            url=load_test.API_URL,
        )

    monkeypatch.setattr(load_test, "run_load_test", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "load_test.py",
            "--url",
            "http://api.example.test/",
            "--duration",
            "1",
            "--workers",
            "2",
            "--batch",
        ],
    )

    load_test.main()
    assert captured == {
        "duration": 1,
        "workers": 2,
        "batch": True,
        "allow_unhealthy": False,
        "url": "http://api.example.test",
    }
