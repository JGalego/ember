"""FastAPI service exposing the EBM solver: POST /solve, /energy, /optimize.

Model weights are demo weights by default: a small MLP encoder/energy/decoder
initialized from a fixed seed at startup, with the hand-specified constraint
energy weighted heavily relative to the (untrained) learned energy head, so
`/solve` returns genuinely useful answers without requiring a trained
checkpoint first. Point `KONA_EBM_CHECKPOINT_<DOMAIN>` (e.g.
`KONA_EBM_CHECKPOINT_SUDOKU`) at a Lightning checkpoint produced by
`scripts/train.py` to serve trained weights for that domain instead.

Run with: uvicorn kona_ebm.api.main:app --reload
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kona_ebm.datasets import available_domains, get_domain
from kona_ebm.datasets.domain import Domain
from kona_ebm.inference import optimize_candidate
from kona_ebm.inference.solve import solve as run_solve
from kona_ebm.models import build_demo_bundle

LATENT_DIM = 32

app = FastAPI(
    title="kona-ebm API",
    description=(
        "Research API for an energy-based reasoning framework over constraint-satisfaction "
        "problems, inspired by publicly discussed energy-based modeling (EBM) concepts and "
        "public, non-technical descriptions of Logical Intelligence's Kona-1. This is an "
        "original implementation, not a reproduction of, or claim of equivalence to, any "
        "proprietary system. See REFERENCES.md in the repository."
    ),
    version="0.1.0",
)


class ModelBundle:
    def __init__(self, domain: Domain) -> None:
        self.domain = domain
        self.encoder, self.energy, self.decoder = build_demo_bundle(domain, latent_dim=LATENT_DIM)

        env_key = f"KONA_EBM_CHECKPOINT_{domain.name.upper()}"
        ckpt_path = os.environ.get(env_key)
        if ckpt_path:
            self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {
            k.removeprefix("encoder."): v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        energy_state = {
            k.removeprefix("energy."): v for k, v in state_dict.items() if k.startswith("energy.")
        }
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
        if energy_state:
            self.energy.load_state_dict(energy_state, strict=False)


@cache
def get_bundle(domain_name: str) -> ModelBundle:
    if domain_name not in available_domains():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown domain '{domain_name}'. Available: {available_domains()}",
        )
    torch.manual_seed(0)
    return ModelBundle(get_domain(domain_name))


class EnergyRequest(BaseModel):
    domain: str
    problem: Any
    candidate: Any


class EnergyResponse(BaseModel):
    energy: float


class OptimizeRequest(BaseModel):
    domain: str
    problem: Any
    method: str = "adam"
    lr: float = 0.1
    max_iters: int = 200
    tol: float = 1e-4
    patience: int = 10


class OptimizeResponse(BaseModel):
    final_energy: float
    energy_history: list[float]
    n_iters: int
    candidate: list[float]


class SolveRequest(BaseModel):
    domain: str
    problem: Any
    method: str = "adam"
    lr: float = 0.1
    max_iters: int = 200
    n_starts: int = 4
    max_restarts: int = 3
    energy_threshold: float = 0.5


class SolveResponse(BaseModel):
    solution: Any
    valid: bool
    violations: int
    final_energy: float
    n_iters: int
    n_restarts: int
    runtime_s: float


@app.get("/domains")
def list_domains() -> list[str]:
    return available_domains()


@app.post("/energy", response_model=EnergyResponse)
def energy_endpoint(req: EnergyRequest) -> EnergyResponse:
    bundle = get_bundle(req.domain)
    domain = bundle.domain
    try:
        problem_t = domain.encode_problem(req.problem).unsqueeze(0)
        candidate_t = domain.encode_solution(req.candidate).unsqueeze(0)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not encode request for domain '{req.domain}': {exc}"
        ) from exc
    with torch.no_grad():
        latent = bundle.encoder(problem_t)
        e = bundle.energy(problem_t, latent, candidate_t)
    return EnergyResponse(energy=e.item())


@app.post("/optimize", response_model=OptimizeResponse)
def optimize_endpoint(req: OptimizeRequest) -> OptimizeResponse:
    bundle = get_bundle(req.domain)
    domain = bundle.domain
    try:
        problem_t = domain.encode_problem(req.problem).unsqueeze(0)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not encode problem for domain '{req.domain}': {exc}"
        ) from exc
    with torch.no_grad():
        latent = bundle.encoder(problem_t)
    init = domain.random_solution(req.problem).unsqueeze(0)
    result = optimize_candidate(
        bundle.energy,
        problem_t,
        latent,
        init,
        method=req.method,
        lr=req.lr,
        max_iters=req.max_iters,
        tol=req.tol,
        patience=req.patience,
    )
    return OptimizeResponse(
        final_energy=result.energy_history[-1],
        energy_history=result.energy_history,
        n_iters=result.n_iters,
        candidate=result.candidate.squeeze(0).tolist(),
    )


@app.post("/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest) -> SolveResponse:
    bundle = get_bundle(req.domain)
    domain = bundle.domain
    try:
        result = run_solve(
            domain,
            req.problem,
            bundle.encoder,
            bundle.energy,
            decoder=bundle.decoder,
            method=req.method,
            lr=req.lr,
            max_iters=req.max_iters,
            n_starts=req.n_starts,
            max_restarts=req.max_restarts,
            energy_threshold=req.energy_threshold,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Solve failed for domain '{req.domain}': {exc}"
        ) from exc
    return SolveResponse(
        solution=result.solution,
        valid=result.valid,
        violations=result.violations,
        final_energy=result.final_energy,
        n_iters=result.n_iters,
        n_restarts=result.n_restarts,
        runtime_s=result.runtime_s,
    )
