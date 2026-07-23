import pytest
import torch

from ember.models.decoder import MLPDecoder, TransformerDecoder, build_decoder

SOLUTION_DIM = 64
LATENT_DIM = 16
BATCH = 4


@pytest.mark.parametrize(
    "kind,extra_kwargs",
    [
        ("mlp", {}),
        ("transformer", {"n_tokens": 8, "token_dim": 16, "n_layers": 1, "n_heads": 2}),
    ],
)
def test_decoder_output_shape_with_latent(kind, extra_kwargs):
    decoder = build_decoder(kind, solution_dim=SOLUTION_DIM, latent_dim=LATENT_DIM, **extra_kwargs)
    candidate = torch.randn(BATCH, SOLUTION_DIM)
    latent = torch.randn(BATCH, LATENT_DIM)
    out = decoder(candidate, latent)
    assert out.shape == (BATCH, SOLUTION_DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_decoder_works_without_latent_context(kind):
    decoder = build_decoder(kind, solution_dim=SOLUTION_DIM, latent_dim=0)
    candidate = torch.randn(BATCH, SOLUTION_DIM)
    out = decoder(candidate)
    assert out.shape == (BATCH, SOLUTION_DIM)


def test_decoder_is_identity_when_correction_is_zeroed():
    # The decoder is `candidate + correction_net(...)`: zeroing the final
    # layer's weight/bias makes the correction exactly 0, so the decoder must
    # reduce to the identity on `candidate`.
    decoder = MLPDecoder(solution_dim=SOLUTION_DIM, latent_dim=0, hidden_dims=(32,))
    last_linear = decoder.net[-1]
    with torch.no_grad():
        last_linear.weight.zero_()
        last_linear.bias.zero_()
    candidate = torch.randn(BATCH, SOLUTION_DIM)
    out = decoder(candidate)
    assert torch.allclose(out, candidate)


def test_build_decoder_rejects_unknown_kind():
    with pytest.raises(KeyError):
        build_decoder("not_a_real_decoder", solution_dim=SOLUTION_DIM)


def test_direct_class_instantiation():
    mlp = MLPDecoder(solution_dim=SOLUTION_DIM, latent_dim=LATENT_DIM)
    transformer = TransformerDecoder(solution_dim=SOLUTION_DIM, latent_dim=LATENT_DIM)
    out1 = mlp(torch.randn(2, SOLUTION_DIM), torch.randn(2, LATENT_DIM))
    out2 = transformer(torch.randn(2, SOLUTION_DIM), torch.randn(2, LATENT_DIM))
    assert out1.shape == out2.shape == (2, SOLUTION_DIM)
