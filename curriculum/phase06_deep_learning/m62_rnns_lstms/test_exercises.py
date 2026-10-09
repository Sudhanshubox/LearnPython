import pytest
import torch
import torch.nn as nn

from exercises import (
    CharLSTM,
    clip_grad_norm,
    input_gradient_norms,
    lstm_cell,
    make_char_dataset,
    rnn_cell,
    rnn_forward,
    sample,
    train_char_model,
)

TEXT = "the quick brown fox jumps over the lazy dog. " * 40


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_rnn_cell():
    x, h = torch.randn(3, 4, generator=gen(0)), torch.randn(3, 5, generator=gen(1))
    W_xh, W_hh, b = torch.randn(4, 5, generator=gen(2)), torch.randn(5, 5, generator=gen(3)), torch.randn(5, generator=gen(4))
    assert torch.allclose(rnn_cell(x, h, W_xh, W_hh, b), torch.tanh(x @ W_xh + h @ W_hh + b))


def test_rnn_forward_matches_pytorch():
    torch.manual_seed(0)
    ref = nn.RNN(4, 6, batch_first=True)
    x = torch.randn(2, 7, 4, generator=gen(1))
    h0 = torch.randn(2, 6, generator=gen(2))
    out, h = rnn_forward(x, h0, ref.weight_ih_l0.T, ref.weight_hh_l0.T, ref.bias_ih_l0 + ref.bias_hh_l0)
    ref_out, ref_h = ref(x, h0[None])
    assert out.shape == (2, 7, 6) and h.shape == (2, 6)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(h, ref_h[0], atol=1e-5)


def test_lstm_cell_matches_pytorch():
    torch.manual_seed(0)
    ref = nn.LSTMCell(3, 5)
    x = torch.randn(4, 3, generator=gen(1))
    h, c = torch.randn(4, 5, generator=gen(2)), torch.randn(4, 5, generator=gen(3))
    h2, c2 = lstm_cell(x, h, c, ref.weight_ih.T, ref.weight_hh.T, ref.bias_ih + ref.bias_hh)
    ref_h, ref_c = ref(x, (h, c))
    assert torch.allclose(h2, ref_h, atol=1e-5)
    assert torch.allclose(c2, ref_c, atol=1e-5)


def test_clip_grad_norm():
    a = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    unused = torch.zeros(3, requires_grad=True)
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])
    total = clip_grad_norm([a, b, unused], 1.0)
    assert total == pytest.approx(5.0)
    assert torch.allclose(a.grad, torch.tensor([0.6, 0.0]), atol=1e-5)
    assert torch.allclose(b.grad, torch.tensor([0.8]), atol=1e-5)
    a.grad, b.grad = torch.tensor([0.3, 0.0]), torch.tensor([0.4])
    assert clip_grad_norm([a, b], 1.0) == pytest.approx(0.5)
    assert torch.allclose(a.grad, torch.tensor([0.3, 0.0])), "don't scale gradients below the limit"


def test_input_gradient_norms():
    torch.manual_seed(0)
    rnn = nn.RNN(4, 8, batch_first=True)
    x = torch.randn(1, 6, 4, generator=gen(1))
    norms = input_gradient_norms(rnn, x)
    assert len(norms) == 6 and all(isinstance(v, float) for v in norms)
    assert x.grad is None
    xr = x.clone().requires_grad_(True)
    rnn(xr)[0][:, -1].sum().backward()
    assert norms[2] == pytest.approx(xr.grad[0, 2].norm().item())


def test_vanishing_gradients_experiment():
    x = torch.randn(1, 50, 4, generator=gen(0))
    torch.manual_seed(0)
    rnn = nn.RNN(4, 32, batch_first=True)
    r = input_gradient_norms(rnn, x)
    assert r[0] / r[-1] < 1e-6, "a vanilla RNN forgets the start of a 50-step sequence"
    torch.manual_seed(0)
    lstm = nn.LSTM(4, 32, batch_first=True)
    with torch.no_grad():
        lstm.bias_hh_l0[32:64] = 3.0          # forget-gate bias: remember by default
    l = input_gradient_norms(lstm, x)
    assert l[0] / l[-1] > 1e-2, "the LSTM's cell state keeps the gradient alive"


def test_make_char_dataset():
    vocab, stoi, X, Y = make_char_dataset("hello world", 3)
    assert vocab == sorted(set("hello world")) and stoi["h"] == vocab.index("h")
    assert X.dtype == torch.int64 and X.shape == (3, 3) and Y.shape == (3, 3)
    decode = lambda row: "".join(vocab[i] for i in row)
    assert [decode(r) for r in X] == ["hel", "lo ", "wor"]
    assert [decode(r) for r in Y] == ["ell", "o w", "orl"]


def test_char_lstm_shapes():
    torch.manual_seed(0)
    model = CharLSTM(10, embed_dim=8, hidden_size=12)
    assert isinstance(model.embed, nn.Embedding) and isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.head, nn.Linear)
    x = torch.randint(0, 10, (3, 5), generator=gen(1))
    logits, (h, c) = model(x)
    assert logits.shape == (3, 5, 10) and h.shape == (1, 3, 12)
    full, _ = model(x[:, :4])
    first, state = model(x[:, :2])
    rest, _ = model(x[:, 2:4], state)
    assert torch.allclose(torch.cat([first, rest], dim=1), full, atol=1e-5), "carry the state"


@pytest.fixture(scope="module")
def trained():
    return train_char_model(TEXT, epochs=30)


@pytest.mark.timeout(120)
def test_training(trained):
    model, vocab, stoi, losses = trained
    assert len(losses) == 30
    assert losses[0] > 2.0 and losses[-1] < 0.2
    again = train_char_model(TEXT, epochs=2)[3]
    assert again == train_char_model(TEXT, epochs=2)[3], "seed everything"


@pytest.mark.timeout(60)
def test_sampling(trained):
    model, vocab, stoi, _ = trained
    model.train()
    greedy = sample(model, stoi, vocab, "the ", 60, temperature=0)
    assert not model.training
    assert len(greedy) == 64 and greedy.startswith("the ")
    assert greedy == (TEXT * 2)[:64]
    s1 = sample(model, stoi, vocab, "the ", 40, 1.0, gen(5))
    s2 = sample(model, stoi, vocab, "the ", 40, 1.0, gen(5))
    assert s1 == s2 and len(s1) == 44 and set(s1) <= set(vocab)
    hot = sample(model, stoi, vocab, "the ", 300, 5.0, gen(1))
    assert hot[4:] not in TEXT * 10, "a very high temperature produces random-looking text"
