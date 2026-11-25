"""Generate SMILES from DreaMS spectrum embeddings.

The DreaMS encoder is discriminative: its embedding cannot be inverted into a structure. This
module trains the smallest honest decoder - a character-level LSTM seeded with the spectrum
embedding - on labelled spectra, so an unseen spectrum can propose a SMILES. Quality is judged
against the nearest-neighbour retrieval baseline from the same embeddings: if the decoder cannot
beat "copy the closest training spectrum's molecule", the embedding carries no generative signal
beyond retrieval.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch import nn
from torch.nn import functional as F

# Monoisotopic mass and maximum valence of every atom a SMILES character can introduce.
ATOMS = {"C": (12.0, 4), "N": (14.003074, 3), "O": (15.994915, 2), "S": (31.972071, 6),
         "P": (30.973762, 5), "F": (18.998403, 1), "Cl": (34.968853, 1), "Br": (78.918338, 1),
         "I": (126.904473, 1), "B": (11.009305, 3), "Si": (28.976434, 4),
         "c": (12.0, 3), "n": (14.003074, 3), "o": (15.994915, 2), "s": (31.972071, 4)}
H_MASS = 1.007825


class SmilesGenerator(nn.Module):
    """Character-level SMILES decoder conditioned on a spectrum embedding.

    The embedding is projected into the LSTM stream as the first "token"; the network then
    predicts the next character until it emits <end>.
    """

    def __init__(self, dim: int, vocab: list[str], hidden: int = 256, layers: int = 2):
        super().__init__()
        self.vocab = vocab
        self.end = len(vocab)
        self.embed = nn.Embedding(self.end + 1, hidden)
        self.proj = nn.Linear(dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, self.end + 1)

    def loss(self, z: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: logits at position i predict character i+1 (END after the last)."""
        x = torch.cat([self.proj(z)[:, None, :], self.embed(chars)], dim=1)
        out, _ = self.lstm(x)
        logits = self.out(out)
        target = torch.cat([chars, torch.full((len(chars), 1), self.end, device=chars.device)], dim=1)
        return F.cross_entropy(logits.reshape(-1, self.end + 1), target.reshape(-1))

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, max_len: int = 120, temperature: float = 0.0) -> list[str]:
        """Greedy (temperature 0) or sampled generation for a batch of embeddings."""
        h = c = None
        x = self.proj(z)[:, None, :]
        chars, done = [[] for _ in range(len(z))], torch.zeros(len(z), dtype=torch.bool, device=z.device)
        for _ in range(max_len):
            out, (h, c) = self.lstm(x, None if h is None else (h, c))
            logits = self.out(out[:, -1])
            nxt = logits.argmax(1) if temperature <= 0 else \
                torch.multinomial(F.softmax(logits / temperature, 1), 1).squeeze(1)
            for i in range(len(z)):
                if not done[i] and int(nxt[i]) != self.end:
                    chars[i].append(self.vocab[int(nxt[i])])
            done |= nxt == self.end
            x = self.embed(nxt[:, None])
            if bool(done.all()):
                break
        return ["".join(s) for s in chars]

    @torch.inference_mode()
    def generate_mass_constrained(self, z, target_mass: float | np.ndarray, tol: float = 0.05,
                                  beams: int = 4, max_len: int = 120) -> list[tuple[str, bool, float]]:
        """Beam search in which no prefix can end up heavier than the MS1 precursor allows.

        `target_mass` is the NEUTRAL monoisotopic mass (precursor minus the adduct proton). Each beam
        carries the heavy-atom mass it has emitted and the free valence left on those atoms, so the
        exact mass of any molecule that beam can still complete is bracketed by
        [heavy, heavy + free_valence * H]. A beam survives while that bracket still admits the target:
        it is dropped once it is already too heavy, or once even the atoms left in the length budget
        could not make it heavy enough. The returned molecule is the highest-scoring completed beam
        that RDKit parses and whose exact mass is within tol; matched is False when nothing passed, in
        which case an empty string is returned so the caller can tell a hit from a guess.

        shortcut: bracketed atoms other than H, and the rare As/Se, are not counted, so the bracket is
        loose for them; RDKit still decides the final verdict.
        """
        z = torch.as_tensor(z, device=self.proj.weight.device)
        n = len(z)
        target = np.full(n, float(target_mass)) if np.isscalar(target_mass) else np.asarray(target_mass)
        # An atom token adds its mass and its free valence minus the implicit bond to the previous
        # atom; a bond token removes the valence it consumes. A ring digit appears twice per ring.
        mass_of = np.zeros(self.end + 1)
        free_of = np.zeros(self.end + 1)
        for i, ch in enumerate(self.vocab):
            if ch in ATOMS:
                mass_of[i], free_of[i] = ATOMS[ch][0], ATOMS[ch][1] - 2
            elif ch in "=#":
                free_of[i] = -2
            elif ch.isdigit():
                free_of[i] = -1
            elif ch == "H":
                mass_of[i], free_of[i] = H_MASS, -1
        code = {c: i for i, c in enumerate(self.vocab)}
        # "Cl", "Br" and "Si" are two characters: the second one rewrites the atom already open.
        fixes = {("C", "l"): (ATOMS["Cl"][0] - ATOMS["C"][0], -3),
                 ("B", "r"): (ATOMS["Br"][0] - ATOMS["B"][0], -2),
                 ("S", "i"): (ATOMS["Si"][0] - ATOMS["S"][0], -2)}
        heaviest = max(m for m, _ in ATOMS.values())
        # A prefix that cannot become a valid SMILES is dropped whatever its mass is: the mass rule
        # alone cannot stop a beam from looping on massless tokens such as "----".
        is_bond = np.zeros(self.end + 1, bool)
        ring_bit = np.zeros(self.end + 1, int)
        dparen = np.zeros(self.end + 1, int)
        dbracket = np.zeros(self.end + 1, int)
        for i, ch in enumerate(self.vocab):
            if ch in "-=#":
                is_bond[i] = True
            elif ch.isdigit():
                ring_bit[i] = 1 << int(ch)
            elif ch == "(":
                dparen[i] = 1
            elif ch == ")":
                dparen[i] = -1
            elif ch == "[":
                dbracket[i] = 1
            elif ch == "]":
                dbracket[i] = -1

        x = self.proj(z)[:, None, :]
        h = c = None
        rows = list(range(n))  # one live beam per row to start; the beam list stays compact
        chars = [[] for _ in rows]
        mass, free, score = np.zeros(len(rows)), np.full(len(rows), 2.0), np.zeros(len(rows))
        last_atom = np.full(len(rows), -1)
        prev = np.full(len(rows), -1)
        paren = np.zeros(len(rows), int)
        bracket = np.zeros(len(rows), int)
        ring_open = np.zeros(len(rows), int)
        ring_closed = np.zeros(len(rows), int)
        best = [None] * n  # (matched, score, smiles, exact mass)
        parsed: dict[str, float | None] = {}

        for step in range(max_len):
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)
            logp = torch.log_softmax(self.out(out[:, -1]), 1).cpu().numpy()
            # END is always a candidate so a beam can close when the greedy continuation is infeasible.
            top = np.concatenate([np.argsort(logp, 1)[:, -beams:],
                                  np.full((len(rows), 1), self.end, dtype=int)], axis=1)
            pidx = np.repeat(np.arange(len(rows)), top.shape[1])
            tok = top.reshape(-1)
            dm, df = mass_of[tok], free_of[tok]
            for (a, b), (ma, fa) in fixes.items():
                if a in code and b in code:
                    hit = (tok == code[b]) & (last_atom[pidx] == code[a])
                    dm[hit] += ma
                    df[hit] += fa
            row = np.array(rows)[pidx]
            new_mass, new_free, new_score = mass[pidx] + dm, free[pidx] + df, score[pidx] + logp[pidx, tok]
            new_paren = paren[pidx] + dparen[tok]
            new_bracket = bracket[pidx] + dbracket[tok]
            bit = ring_bit[tok]
            new_open = ring_open[pidx] | (bit & ~ring_open[pidx])
            new_closed = ring_closed[pidx] | (bit & ring_open[pidx])
            open_p, close_p = code.get("(", -1), code.get(")", -1)
            prev = prev[pidx]
            syntax = (new_paren >= 0) & (new_bracket >= 0) & ((ring_closed[pidx] & bit) == 0) & \
                    ~((tok == open_p) & (is_bond[prev] | (prev == open_p))) & \
                    ~((tok == close_p) & (prev == open_p)) & ~(is_bond[tok] & is_bond[prev])

            # A closed beam can only match if the target sits inside its mass bracket, so RDKit is
            # only asked about the plausible ones.
            for i in np.flatnonzero(syntax & (tok == self.end) & (new_mass <= target[row] + tol) & (
                    new_mass + new_free * H_MASS >= target[row] - tol)):
                smi = "".join(chars[pidx[i]])
                if smi not in parsed:
                    mol = Chem.MolFromSmiles(smi)
                    parsed[smi] = float(rdMolDescriptors.CalcExactMolWt(mol)) if mol else None
                exact = parsed[smi]
                if exact is None:
                    continue
                r = row[i]
                matched = abs(exact - float(target[r])) <= tol
                if best[r] is None or (matched, new_score[i]) > best[r][:2]:
                    best[r] = (matched, float(new_score[i]), smi, exact)

            # A prefix can still grow, so only "already too heavy", "can never get heavy enough" and
            # "cannot be a SMILES at all" are sound pruning rules.
            keep = syntax & (tok != self.end) & (new_mass <= target[row] + tol) & (
                new_mass + new_free * H_MASS + (max_len - step) * heaviest >= target[row] - tol)
            cand = np.flatnonzero(keep)
            if cand.size == 0:
                break
            order = cand[np.lexsort((-new_score[cand], row[cand]))]
            used, sel = {}, []
            for idx in order:
                r = row[idx]
                if used.get(r, 0) < beams:
                    used[r] = used.get(r, 0) + 1
                    sel.append(idx)
            rows = [row[i] for i in sel]
            chars = [chars[pidx[i]] + [self.vocab[tok[i]]] for i in sel]
            mass, free, score = new_mass[sel], new_free[sel], new_score[sel]
            paren, bracket, ring_open, ring_closed, prev = (new_paren[sel], new_bracket[sel],
                                                            new_open[sel], new_closed[sel], tok[sel])
            last_atom = np.array([tok[i] if self.vocab[tok[i]] in ATOMS else last_atom[pidx[i]] for i in sel])
            parents = pidx[sel]
            x = self.embed(torch.tensor([tok[i] for i in sel], dtype=torch.long, device=z.device)[:, None])
            h, c = h[:, parents, :], c[:, parents, :]

        return [(b[2], b[0], b[3]) if b else ("", False, 0.0) for b in best]
