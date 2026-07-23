import pytest

pytest.importorskip("httpx")  # FastAPI's TestClient needs httpx installed

from fastapi.testclient import TestClient

from kona_ebm.api.main import app
from kona_ebm.datasets import get_domain


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_list_domains(client):
    r = client.get("/domains")
    assert r.status_code == 200
    assert set(r.json()) == {"sudoku", "sat", "graph_coloring", "maze"}


def test_energy_endpoint(client):
    domain = get_domain("graph_coloring")  # server bundle uses domain defaults; must match
    inst = domain.generate(1, seed=0)[0]
    r = client.post(
        "/energy",
        json={"domain": "graph_coloring", "problem": inst.problem, "candidate": inst.solution},
    )
    assert r.status_code == 200
    assert "energy" in r.json()


def test_optimize_endpoint(client):
    domain = get_domain("graph_coloring")  # server bundle uses domain defaults; must match
    inst = domain.generate(1, seed=0)[0]
    r = client.post(
        "/optimize", json={"domain": "graph_coloring", "problem": inst.problem, "max_iters": 20}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["energy_history"]) == body["n_iters"]
    assert len(body["candidate"]) == domain.solution_dim


def test_solve_endpoint(client):
    domain = get_domain("graph_coloring")  # server bundle uses domain defaults; must match
    inst = domain.generate(1, seed=0)[0]
    r = client.post(
        "/solve",
        json={"domain": "graph_coloring", "problem": inst.problem, "max_iters": 100, "n_starts": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "violations" in body and "valid" in body


def test_unknown_domain_returns_404(client):
    r = client.post("/energy", json={"domain": "not_a_domain", "problem": [], "candidate": []})
    assert r.status_code == 404
