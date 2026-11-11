import pytest
from fastapi.testclient import TestClient
from triage.main import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAGE_DB", str(tmp_path / "tickets.db"))
    with TestClient(app) as c:
        yield c

def test_health(client):
    assert client.get('/health').json() == {"status": "ok"}

def test_create_get_and_list(client):
    response = client.post('/tickets', json={'text': 'invoice payment failed'})
    assert response.status_code == 201
    ticket = response.json()
    assert ticket['predicted_category'] == 'billing'
    assert ticket['status'] == 'pending'
    assert 0 <= ticket['confidence'] <= 1
    assert client.get(f"/tickets/{ticket['id']}").json() == ticket
    assert len(client.get('/tickets?status=pending').json()) == 1
    assert client.get('/tickets?status=reviewed').json() == []

def test_human_review_is_final_and_preserves_prediction(client):
    ticket = client.post('/tickets', json={'text': 'password reset'}).json()
    url = f"/tickets/{ticket['id']}/review"
    reviewed = client.post(url, json={'category': 'technical', 'note': 'corrected'}).json()
    assert reviewed['final_category'] == 'technical'
    assert reviewed['predicted_category'] == ticket['predicted_category']
    assert reviewed['status'] == 'reviewed'
    assert reviewed['reviewed_at']
    assert client.post(url, json={'category': 'billing'}).status_code == 409

def test_missing_and_invalid(client):
    assert client.get('/tickets/999').status_code == 404
    assert client.post('/tickets/999/review', json={'category': 'billing'}).status_code == 404
    assert client.post('/tickets', json={'text': '   '}).status_code == 422
    assert client.post('/tickets', json={'text': 'a' * 4001}).status_code == 422
    assert client.get('/tickets?limit=101').status_code == 422
    assert client.get('/tickets?status=nope').status_code == 422
    assert client.post('/tickets/1/review', json={'category': 'other'}).status_code == 422

def test_persists_across_clients(client):
    ticket = client.post('/tickets', json={'text': 'login password reset'}).json()
    with TestClient(app) as second:
        assert second.get(f"/tickets/{ticket['id']}").status_code == 200

def test_low_confidence_flag(client):
    ticket = client.post('/tickets', json={'text': 'xyzzy unrelated words'}).json()
    assert ticket['requires_review'] is True
