import base64
from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_generate_image_requires_auth(client: TestClient) -> None:
    response = client.post("/v1/generate/image", json={"prompt": "test"})
    assert response.status_code in (401, 422)


def test_generate_image_returns_base64_png(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch,
) -> None:
    from any_llm.gateway.routes import image as image_route

    class _FakeInlineData:
        def __init__(self) -> None:
            self.mime_type = "image/png"
            self.data = b"fake-png-bytes"

    class _FakePart:
        def __init__(self) -> None:
            self.inline_data = _FakeInlineData()
            self.text = "caption"
            self.thought = False

    class _FakeThoughtPart:
        def __init__(self) -> None:
            self.text = "thinking"
            self.thought = True

    class _FakeResponse:
        def __init__(self) -> None:
            self.parts = [_FakeThoughtPart(), _FakePart()]

    class _FakeModels:
        def generate_content(self, *, model: str, contents, config) -> _FakeResponse:  # noqa: ANN001
            assert model
            assert contents
            assert config is not None
            assert config["response_modalities"] == ["Text", "Image"]
            return _FakeResponse()

    class _FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key
            self.models = _FakeModels()

    monkeypatch.setattr(image_route, "validate_user_credit", lambda _db, _user_id: None)
    monkeypatch.setattr(image_route, "_get_gemini_api_key", lambda _config: "test-api-key")
    monkeypatch.setattr(
        image_route,
        "genai",
        SimpleNamespace(
            Client=_FakeClient,
            types=SimpleNamespace(
                GenerateContentConfig=lambda **kwargs: kwargs,
                ImageConfig=lambda **kwargs: kwargs,
            ),
        ),
    )

    response = client.post(
        "/v1/generate/image",
        json={"prompt": "A cute cat"},
        headers=api_key_header,
    )
    assert response.status_code == 200
    data = response.json()

    assert data["mimeType"] == "image/png"
    assert data["base64"] == base64.b64encode(b"fake-png-bytes").decode("utf-8")
    assert data["texts"] == ["caption"]
    assert data["thoughts"] == ["thinking"]
