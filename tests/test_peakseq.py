"""The tokeniser keeps the strongest peaks in ascending order, dropping blanks only real peaks,
and the JEPA loss goes down when the encoder can restore the peaks it was not shown."""
import numpy as np
import torch

from sna.peakseq import PeakEncoder, peak_tokens


def test_peak_tokens_keep_the_strongest_peaks_sorted():
    mz = [100.0, 101.0, 200.0, 50.0, 51.0]
    inten = [1.0, 5.0, 2.0, 9.0, 3.0]
    tokens = peak_tokens(mz, inten, [0, len(mz)], bins=1000, max_peaks=3)
    assert tokens.tolist() == [[50, 51, 101]]


def test_padding_is_zero_and_drop_only_blanks_real_peaks():
    mz = [10.0, 20.0, 30.0]
    tokens = peak_tokens(mz, [1, 1, 1], [0, 3], bins=1000, max_peaks=5)
    assert tokens.tolist() == [[10, 20, 30, 0, 0]]
    model = PeakEncoder(bins=1000, hidden=32, layers=1, dim=16, max_peaks=5)
    dropped = model.drop_peaks(torch.tensor(tokens), fraction=1.0)
    assert dropped.tolist() == [[0, 0, 0, 0, 0]]


def test_jepa_loss_decreases_with_training():
    torch.manual_seed(0)
    tokens = torch.randint(1, 1000, (64, 32), device="cuda")
    model = PeakEncoder(hidden=64, layers=2, dim=32, max_peaks=32).cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    first = model.jepa_loss(tokens).item()
    for _ in range(30):
        obj = model.jepa_loss(tokens)
        opt.zero_grad(); obj.backward(); opt.step()
        model.update_teacher()
    assert model.jepa_loss(tokens).item() < first
