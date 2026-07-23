import pytest
import torch

from kona_ebm.models.encoder import GNNEncoder, MLPEncoder, TransformerEncoder, build_encoder

PROBLEM_DIM = 100
LATENT_DIM = 16
BATCH = 4


@pytest.mark.parametrize(
    "kind,extra_kwargs",
    [
        ("mlp", {}),
        ("transformer", {"n_tokens": 8, "token_dim": 16, "n_layers": 1, "n_heads": 2}),
        ("gnn", {"n_nodes": 10, "node_dim": 8, "n_layers": 2}),
    ],
)
def test_encoder_output_shape(kind, extra_kwargs):
    encoder = build_encoder(kind, problem_dim=PROBLEM_DIM, latent_dim=LATENT_DIM, **extra_kwargs)
    problem = torch.randn(BATCH, PROBLEM_DIM)
    latent = encoder(problem)
    assert latent.shape == (BATCH, LATENT_DIM)
    assert torch.isfinite(latent).all()


@pytest.mark.parametrize("kind", ["mlp", "transformer", "gnn"])
def test_encoder_gradients_flow(kind):
    encoder = build_encoder(kind, problem_dim=PROBLEM_DIM, latent_dim=LATENT_DIM)
    problem = torch.randn(BATCH, PROBLEM_DIM, requires_grad=True)
    latent = encoder(problem)
    latent.sum().backward()
    assert problem.grad is not None
    assert torch.isfinite(problem.grad).all()
    assert any(p.grad is not None for p in encoder.parameters())


def test_gnn_encoder_accepts_explicit_adjacency():
    n_nodes = 6
    encoder = GNNEncoder(
        problem_dim=PROBLEM_DIM, latent_dim=LATENT_DIM, n_nodes=n_nodes, node_dim=8, n_layers=2
    )
    problem = torch.randn(BATCH, PROBLEM_DIM)
    adjacency = torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)
    latent = encoder(problem, adjacency=adjacency)
    assert latent.shape == (BATCH, LATENT_DIM)


def test_build_encoder_rejects_unknown_kind():
    with pytest.raises(KeyError):
        build_encoder("not_a_real_encoder", problem_dim=PROBLEM_DIM, latent_dim=LATENT_DIM)


def test_direct_class_instantiation():
    mlp = MLPEncoder(problem_dim=PROBLEM_DIM, latent_dim=LATENT_DIM)
    transformer = TransformerEncoder(problem_dim=PROBLEM_DIM, latent_dim=LATENT_DIM)
    assert mlp.latent_dim == LATENT_DIM
    assert transformer.latent_dim == LATENT_DIM
