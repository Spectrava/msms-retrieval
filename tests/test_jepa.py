"""The JEPA latent carries peaks the encoder was never shown."""
import numpy as np
import torch

from sna.jepa import JEPA, linear_probe, mask_windows


def test_mask_windows_blanks_about_the_requested_fraction():
    blank = (mask_windows(torch.ones(16, 1000), windows=3, fraction=0.2) == 0).sum(1) / 1000
    assert 0.15 <= float(blank.mean()) <= 0.21, f"blanked {float(blank.mean()):.3f} of the range"


def test_latent_recovers_the_hidden_peaks():
    """Each spectrum is a distinct set of spikes; the encoder only ever sees windows of it removed,
    so beating the masked input at reconstructing the whole spectrum means the latent filled them in."""
    rng = np.random.default_rng(0)
    x = np.zeros((64, 1000), dtype=np.float32)
    for i, spikes in enumerate(rng.integers(0, 1000, size=(64, 12))):
        x[i, spikes] = 1.0
    x = torch.tensor(x, device="cuda")

    model = JEPA(1000, hidden=256, dim=128).cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(40):
        obj = model.loss(x, mask_windows(x, 3, 0.2))
        opt.zero_grad(); obj.backward(); opt.step()
        model.update_teacher()

    with torch.no_grad():
        masked = mask_windows(x, 3, 0.2)
        rec = linear_probe(model.latent(x), x)
        cos = lambda a, b: float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-9)).mean())
        latent, input_ = cos(rec, x), cos(masked, x)
    assert latent > input_, f"latent {latent:.3f} is no better than the masked input {input_:.3f}"
