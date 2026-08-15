import torch

from shared.model import (
    DecoderTransformerClassifier,
)


def test_decoder_classifier_shapes():

    model = (
        DecoderTransformerClassifier(

            vocab_size=100,

            max_len=16,

            d_model=32,

            n_heads=4,

            n_layers=2,

            d_ff=64,
        )
    )

    ids = torch.randint(
        0,
        100,
        (4, 16),
    )

    mask = torch.ones(
        4,
        16,
        dtype=torch.long,
    )

    output = model(
        ids,
        mask,
    )

    assert output.shape == (
        4,
        4,
    )


def test_padding_is_supported():

    model = (
        DecoderTransformerClassifier(

            vocab_size=100,

            max_len=8,

            d_model=32,

            n_heads=4,

            n_layers=1,

            d_ff=64,
        )
    )

    ids = torch.randint(
        0,
        100,
        (2, 8),
    )

    mask = torch.tensor(

        [
            [
                1, 1, 1, 1,
                0, 0, 0, 0,
            ],

            [
                1, 1, 1, 0,
                0, 0, 0, 0,
            ],
        ]
    )

    output = model(
        ids,
        mask,
    )

    assert torch.isfinite(
        output
    ).all()


def test_forward_pass_produces_finite_logits():

    model = DecoderTransformerClassifier(
        vocab_size=100,
        max_len=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
    )

    ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    output = model(ids, mask)

    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()