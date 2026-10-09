"""m62 exercises: RNN and LSTM cells from scratch, and a character-level language model.

Shapes are batch-first: x is (B, T, D), hidden states are (B, H).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. One vanilla RNN step: tanh(x_t @ W_xh + h @ W_hh + b).
def rnn_cell(x_t, h, W_xh, W_hh, b):
    raise NotImplementedError


# 2. Run rnn_cell over every time step. Return (outputs (B, T, H), final hidden state (B, H)).
def rnn_forward(x, h0, W_xh, W_hh, b):
    raise NotImplementedError


# 3. One LSTM step (README section 3). W_x is (D, 4H), W_h is (H, 4H), b is (4H,), and the
#    gate pre-activations are split into four chunks in the order i, f, g, o.
#    Return (h_new, c_new).
def lstm_cell(x_t, h, c, W_x, W_h, b):
    raise NotImplementedError


# 4. Clip gradients by their total norm, IN PLACE (README section 2). Skip parameters whose
#    .grad is None. Return the total norm BEFORE clipping, as a float.
def clip_grad_norm(params, max_norm):
    raise NotImplementedError


# 5. How strongly the LAST output depends on each input step. model is an nn.RNN or nn.LSTM
#    with batch_first=True, x is (1, T, D). Compute out, _ = model(x) on a copy of x that
#    requires grad, backpropagate out[:, -1].sum(), and return the list of the gradient
#    norms for each time step t (the norm of x.grad[0, t]) as floats.
def input_gradient_norms(model, x):
    raise NotImplementedError


# 6. Prepare a character dataset: vocab = sorted unique characters, stoi = {char: index},
#    and the text cut into n = (len(text) - 1) // seq_len non-overlapping chunks:
#    X[k] = ids[k*seq_len : (k+1)*seq_len], Y[k] = the same span shifted right by one.
#    Return (vocab, stoi, X, Y) with X and Y int64 tensors of shape (n, seq_len).
def make_char_dataset(text, seq_len):
    raise NotImplementedError


# 7. Embedding(vocab_size, embed_dim) -> LSTM(embed_dim, hidden_size, batch_first=True)
#    -> Linear(hidden_size, vocab_size), stored as self.embed, self.lstm and self.head.
#    forward(x, state=None) takes int64 ids (B, T) and returns (logits (B, T, V), new_state).
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_size=64):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, state=None):
        raise NotImplementedError


# 8. Train a CharLSTM on text: torch.manual_seed(seed) before creating the model, Adam(lr),
#    each epoch a random permutation of the chunks (torch.randperm with a generator seeded
#    with seed) in mini-batches, cross-entropy over all positions, and clip_grad_norm with
#    max_norm before each optimizer step.
#    Return (model, vocab, stoi, losses) where losses has the mean loss per epoch
#    (weighted by batch size).
def train_char_model(text, seq_len=32, epochs=30, lr=1e-2, batch_size=16, max_norm=1.0, seed=0):
    raise NotImplementedError


# 9. Generate n characters after the prompt `start` (README section 5). Run the model over
#    the prompt once, then feed one character at a time, carrying the LSTM state.
#    temperature == 0 means greedy (argmax); otherwise sample with
#    torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=generator).
#    Use eval mode and no gradients. Return the prompt plus the generated characters.
def sample(model, stoi, vocab, start, n, temperature=1.0, generator=None):
    raise NotImplementedError
