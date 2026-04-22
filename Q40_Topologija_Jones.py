#!/usr/bin/env python3

"""
Q40 Topologija (ne-anyonska) — knot invariants / Jones polinomijal,
Temperley-Lieb algebra, Kauffman bracket — čisto kvantno.

Paradigma:
  Jones polinomijal V(L; t) je topološki invarijant orijentisanog linka L.
  Kauffman bracket ⟨L⟩ je varijanta bez orijentacije, definisana na dijagramu
  preko pravila:
        ⟨L⟩ = A · ⟨L_0⟩ + A^{-1} · ⟨L_∞⟩       (skein relacija)
        ⟨O⟩ = 1
        ⟨L ⊔ O⟩ = δ · ⟨L⟩,  δ = −A² − A^{-2}
  Jones polinomijal se dobija iz ⟨·⟩ uz t = A^{-4} i Markov trace preko
  zatvaranja pletenja (braid closure).

Braid grupa B_n i Temperley-Lieb algebra TL_n:
  B_n je generisana σ_1..σ_{n-1} sa Artin relacijama:
        σ_k σ_{k+1} σ_k = σ_{k+1} σ_k σ_{k+1}
        σ_k σ_m = σ_m σ_k   za |k-m| ≥ 2
  TL_n algebra je generisana e_1..e_{n-1} sa:
        e_k² = δ · e_k
        e_k e_{k±1} e_k = e_k
        e_k e_m = e_m e_k  za |k-m| ≥ 2
  Kauffman bracket reprezentacija B_n → TL_n:
        ρ(σ_k) = A · I + A^{-1} · e_k        (i inverz: A^{-1}·I + A·e_k)

Ne-anyonska realizacija na spin-½ lancu (NQ qubita = NQ+1 pramenova):
  e_k deluje na susedne qubit-ove (k, k+1) kao projektor na singlet,
  skaliran da ispuni TL relaciju e_k² = δ·e_k:
        e_k = √2 · |ψ⁻⟩⟨ψ⁻|,   |ψ⁻⟩ = (|01⟩ − |10⟩)/√2
        δ = −A² − A^{-2} (za Kauffman) — u spin-½ "XX"-rep, efektivna vrednost
        se apsorbuje u normalizaciju e_k. Za unitarni σ_k uzima se:
        A = e^(i·3π/8)   ⟹  Re(A²) = cos(3π/4) = −√2/2
  Provera unitarnosti σ_k = A·I + A^{-1}·e_k:
        σ_k σ_k† = |A|² I + (A/A* + A*/A) e_k + |A|^{-2} · e_k²
                 = I + 2Re(A²)·e_k + √2·e_k
                 = I + (−√2 + √2)·e_k = I   ✓

Mapiranje na loto:
  Za svaku poziciju i konstruiše se pleteno slovo w(j_target) od 10 generatora:
        w = σ_{0}^{ε_0} σ_{1}^{ε_1} ... σ_{4}^{ε_4} σ_{4}^{ε_4} ... σ_{0}^{ε_0}
  gde je ε_k = +1 ako je bit k od j_target == 0, inače ε_k = −1. Pleteno slovo
  je deterministička funkcija j_target — topološki kodira strukturalnu cilj-j.

  Inicijalno stanje |ψ_init⟩ = ⊗_{k=0}^{NQ-1} RY(θ_k)|0⟩, sa θ_k biased:
        θ_k = 0.7π   ako je bit k od j_target == 1
        θ_k = 0.3π   inače
  Dakle |ψ_init⟩ ima dominantnu amplitudu blizu |j_target⟩, ali sa rasporedom.

  Braid unitarna:  U_b = ρ(w) — primena 2-qubit σ/σ^{-1} gejtova redom.
  Finalno stanje:  |ψ_fin⟩ = U_b |ψ_init⟩.
  Born sempling iz P(j) = |⟨j|ψ_fin⟩|², maskovano na (num > prev_pick,
  num ∈ [i, i+32]) → num = i + j*.

Dijagnostika (topološki invarijantni pokazatelji):
  • Jones-pseudo-trace: τ_J = ⟨0^NQ| U_b |0^NQ⟩  (ℂ-vrednost)
  • |τ_J|² i arg(τ_J) ≈ informacija o Kauffman bracket invarijantu
  • Fidelity F_j = |⟨j_target| U_b |ψ_init⟩|²  — preživljavanje struktur. targeta

Structural target (bez frekvencije):
        target_i(prev) = prev + (N_MAX − prev) / (N_NUMBERS − i + 2)
        j_target = round(target_i) − i   ∈ [0, 32]

fit: pleteno slovo + TL singlet mešanje = topološki invarijantna
unitarna scattering struktura; prirodno kompatibilno sa 64-dim Hilbert-om.
NQ = 6 qubit-a po poziciji (DIM = 64), reciklirani registar.

Okruženje: Python 3.11.13, qiskit 1.4.4, macOS M1, seed = 39.
CSV = /data/loto7hh_4602_k32.csv
CSV u celini (S̄ kao info).
DeprecationWarning / FutureWarning se gase.
"""


from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass


# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4602_k32.csv")
N_NUMBERS = 7
N_MAX = 39

NQ = 6                              
DIM = 1 << NQ                       # 64
POS_RANGE = 33                      # Num_i ∈ [i, i + 32]

# Kauffman parametar:  A = e^(i · 3π/8)  ⟹  σ_k je unitarna u TL spin-½ rep.
# Jones varijabla: t = A^{-4} = e^(-i · 3π/2) = i.
A_PARAM = np.exp(1j * 3.0 * np.pi / 8.0)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def sort_rows_asc(H: np.ndarray) -> np.ndarray:
    return np.sort(H, axis=1)


# =========================
# Structural target (bez frekvencije)
# =========================
def target_num_structural(position_1based: int, prev_pick: int) -> float:
    denom = float(N_NUMBERS - position_1based + 2)
    return float(prev_pick) + float(N_MAX - prev_pick) / denom


def compute_j_target(position_1based: int, prev_pick: int) -> Tuple[int, float]:
    target = target_num_structural(position_1based, prev_pick)
    j = int(round(target)) - position_1based
    j = max(0, min(POS_RANGE - 1, j))
    return j, target


# =========================
# Temperley-Lieb generator e_k na 2 qubita (4×4 matrica)
#   e_k = √2 · |ψ⁻⟩⟨ψ⁻|,  |ψ⁻⟩ = (|01⟩ − |10⟩)/√2
# =========================
def tl_generator() -> np.ndarray:
    e = np.zeros((4, 4), dtype=np.complex128)
    c = 1.0 / math.sqrt(2.0)
    # e_k [0,0] = e_k[3,3] = 0 (trivijalno na |00⟩, |11⟩)
    e[1, 1] = c
    e[1, 2] = -c
    e[2, 1] = -c
    e[2, 2] = c
    return e


# =========================
# Braid generator σ_k i njegov inverz — Kauffman rep
#   σ_k     = A·I + A^{-1}·e_k
#   σ_k^{-1} = A^{-1}·I + A·e_k       (jer je σ_k unitaran)
# =========================
def braid_gen(inverse: bool) -> np.ndarray:
    I4 = np.eye(4, dtype=np.complex128)
    e = tl_generator()
    if not inverse:
        return A_PARAM * I4 + (1.0 / A_PARAM) * e
    else:
        return (1.0 / A_PARAM) * I4 + A_PARAM * e


SIGMA = braid_gen(inverse=False)
SIGMA_INV = braid_gen(inverse=True)


# =========================
# Inicijalno stanje ⊗_k RY(θ_k)|0⟩, θ_k biased prema bitima j_target
# =========================
def apply_init(qc: QuantumCircuit, qr: QuantumRegister, j_target: int) -> None:
    for k in range(NQ):
        bit = (j_target >> k) & 1
        theta = 0.7 * math.pi if bit == 1 else 0.3 * math.pi
        qc.ry(theta, qr[k])


# =========================
# Pleteno slovo w(j_target): 10 generatora (forward + backward)
#   Za svaki k ∈ {0..4}: ε_k = +1 ako bit_k(j_target) == 0, inače −1
# =========================
def braid_word_from_jt(j_target: int) -> List[Tuple[int, bool]]:
    word: List[Tuple[int, bool]] = []
    for k in range(NQ - 1):
        bit = (j_target >> k) & 1
        inverse = (bit == 1)
        word.append((k, inverse))
    for k in range(NQ - 2, -1, -1):
        bit = (j_target >> k) & 1
        inverse = (bit == 1)
        word.append((k, inverse))
    return word


# =========================
# Braid unitarna: primena slova na qubit registru
# =========================
def apply_braid_word(
    qc: QuantumCircuit,
    qr: QuantumRegister,
    word: List[Tuple[int, bool]],
) -> None:
    for (k, inverse) in word:
        U = SIGMA_INV if inverse else SIGMA
        label = f"sigma_{k+1}" + ("_inv" if inverse else "")
        qc.unitary(U, [qr[k], qr[k + 1]], label=label)


# =========================
# Konstrukcija kola za jednu poziciju
# =========================
def build_circuit(j_target: int) -> QuantumCircuit:
    qr = QuantumRegister(NQ, name="q")
    qc = QuantumCircuit(qr, name="Jones_Q40")
    apply_init(qc, qr, j_target)
    word = braid_word_from_jt(j_target)
    apply_braid_word(qc, qr, word)
    return qc


# =========================
# Jones-pseudo-trace: ⟨0^NQ| U_b |0^NQ⟩
# =========================
def jones_pseudo_trace(j_target: int) -> complex:
    qr = QuantumRegister(NQ, name="q")
    qc = QuantumCircuit(qr, name="Jones_trace")
    word = braid_word_from_jt(j_target)
    apply_braid_word(qc, qr, word)
    sv = Statevector.from_instruction(qc).data
    return complex(sv[0])


# =========================
# Predikcija jedne pozicije
# =========================
def jones_pick_one_position(
    position_1based: int,
    prev_pick: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, complex, float]:
    j_target, target = compute_j_target(position_1based, prev_pick)
    qc = build_circuit(j_target)
    sv = Statevector.from_instruction(qc).data
    probs = np.abs(sv) ** 2

    # Dijagnostika
    tau_J = jones_pseudo_trace(j_target)
    # Fidelity targeta u ψ_fin
    fidelity = float(probs[j_target])

    mask = np.zeros(DIM, dtype=np.float64)
    for j in range(POS_RANGE):
        num = position_1based + j
        if 1 <= num <= N_MAX and num > prev_pick:
            mask[j] = 1.0

    probs_valid = probs * mask
    s = float(probs_valid.sum())
    if s < 1e-15:
        for j in range(POS_RANGE):
            num = position_1based + j
            if 1 <= num <= N_MAX and num > prev_pick:
                return num, j_target, target, tau_J, fidelity
        return (
            max(prev_pick + 1, position_1based),
            j_target,
            target,
            tau_J,
            fidelity,
        )

    probs_valid /= s
    j_sampled = int(rng.choice(DIM, p=probs_valid))
    num = position_1based + j_sampled
    return num, j_target, target, tau_J, fidelity


# =========================
# Autoregresivni run (reciklirani 6-qubit registar)
# =========================
def run_jones_autoregressive() -> List[int]:
    rng = np.random.default_rng(SEED)
    picks: List[int] = []
    prev_pick = 0

    for i in range(1, N_NUMBERS + 1):
        num, j_t, target, tau_J, fid = jones_pick_one_position(
            i, prev_pick, rng
        )
        picks.append(int(num))
        print(
            f"  [pos {i}]  target={target:.3f}  j_target={j_t:2d}  "
            f"|τ_J|={abs(tau_J):.4f}  arg(τ_J)={math.degrees(math.atan2(tau_J.imag, tau_J.real)):+7.2f}°  "
            f"F(j_tgt)={fid:.4f}  num={num:2d}"
        )
        prev_pick = int(num)

    return picks


# =========================
# Main
# =========================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nema CSV: {CSV_PATH}")

    H = load_rows(CSV_PATH)
    H_sorted = sort_rows_asc(H)
    S_bar = float(H_sorted.sum(axis=1).mean())

    a_deg = math.degrees(math.atan2(A_PARAM.imag, A_PARAM.real))

    print("=" * 84)
    print("Q40 Topologija (ne-anyonska) — Jones polinomijal / Kauffman bracket / TL")
    print("=" * 84)
    print(f"CSV:            {CSV_PATH}")
    print(f"Broj redova:    {H.shape[0]}")
    print(f"Qubit budget:   {NQ} po poziciji  (Hilbert dim={DIM})")
    print(
        f"Kauffman A:     |A|={abs(A_PARAM):.4f}  arg(A)={a_deg:+.2f}°  "
        f"(A = e^(i·3π/8))"
    )
    print(f"Jones t:        t = A^(-4) = i")
    print(f"Braid slovo:    10 generatora (forward + backward), ε_k iz bita j_target")
    print(f"TL rep:         e_k = √2 · |ψ⁻⟩⟨ψ⁻|  na (k, k+1)")
    print(f"Srednja suma S̄: {S_bar:.3f}  (CSV info, nije driver)")
    print(f"Seed:           {SEED}")
    print()
    print("Pokretanje Jones (TL + braid + Born sempling) po pozicijama:")

    picks = run_jones_autoregressive()

    n_odd = sum(1 for v in picks if v % 2 == 1)
    gaps = [picks[i + 1] - picks[i] for i in range(N_NUMBERS - 1)]

    print()
    print("=" * 84)
    print("REZULTAT Q40 (NEXT kombinacija)")
    print("=" * 84)
    print(f"Suma:  {sum(picks)}   (S̄={S_bar:.2f})")
    print(f"#odd:  {n_odd}")
    print(f"Gaps:  {gaps}")
    print(f"Predikcija NEXT: {picks}")


if __name__ == "__main__":
    main()



"""
====================================================================================
Q40 Topologija (ne-anyonska) — Jones polinomijal / Kauffman bracket / TL
====================================================================================
CSV:            /data/loto7hh_4602_k32.csv
Broj redova:    4602
Qubit budget:   6 po poziciji  (Hilbert dim=64)
Kauffman A:     |A|=1.0000  arg(A)=+67.50°  (A = e^(i·3π/8))
Jones t:        t = A^(-4) = i
Braid slovo:    10 generatora (forward + backward), ε_k iz bita j_target
TL rep:         e_k = √2 · |ψ⁻⟩⟨ψ⁻|  na (k, k+1)
Srednja suma S̄: 140.509  (CSV info, nije driver)
Seed:           39

Pokretanje Jones (TL + braid + Born sempling) po pozicijama:
  [pos 1]  target=4.875  j_target= 4  |τ_J|=1.0000  arg(τ_J)= +45.00°  F(j_tgt)=0.1031  num= 7
  [pos 2]  target=11.571  j_target=10  |τ_J|=1.0000  arg(τ_J)=+135.00°  F(j_tgt)=0.0126  num=18
  [pos 3]  target=21.500  j_target=19  |τ_J|=1.0000  arg(τ_J)=-135.00°  F(j_tgt)=0.0026  num=21
  [pos 4]  target=24.600  j_target=21  |τ_J|=1.0000  arg(τ_J)=-135.00°  F(j_tgt)=0.0024  num=28
  [pos 5]  target=30.750  j_target=26  |τ_J|=1.0000  arg(τ_J)=-135.00°  F(j_tgt)=0.0111  num=32
  [pos 6]  target=34.333  j_target=28  |τ_J|=1.0000  arg(τ_J)=-135.00°  F(j_tgt)=0.0174  num=38
  [pos 7]  target=38.500  j_target=31  |τ_J|=1.0000  arg(τ_J)= +45.00°  F(j_tgt)=0.0169  num=39

====================================================================================
REZULTAT Q40 (NEXT kombinacija)
====================================================================================
Suma:  183   (S̄=140.51)
#odd:  3
Gaps:  [11, 3, 7, 4, 6, 1]
Predikcija NEXT: [7, 18, 21, 28, 32, 38, 39]
"""



"""
REZULTAT — Q40 Topologija (ne-anyonska) / Jones polinomijal / Kauffman bracket
------------------------------------------------------------------------------
(Popunjava se iz printa main()-a nakon pokretanja.)

Koncept:
  • Čisto kvantno: pleteno slovo = unitarno kolo od 2-qubit
    σ_k / σ_k^{-1} gejtova u TL / Kauffman bracket reprezentaciji. Merenje
    = projekcija statevector-a na computational bazu.
  • Ne-anyonski topološki pristup: braid grupa B_n reprezentacija preko
    Temperley-Lieb algebra e_k = √2·|ψ⁻⟩⟨ψ⁻| na qubit spin-½ lancu.
  • Kauffman parametar A = e^(i·3π/8) osigurava unitarnost σ_k.
  • j_target vodi i inicijalno stanje (RY biased) i pleteno slovo (eksponenti
    ε_k iz bita j_target) — topološki invarijantna scattering struktura.
  • druga paradigma od Q32 (Fibonacci anyoni sa F/R matricama).
  • NQ = 6 qubit-a po poziciji, reciklirani 64-dim registar.
  • deterministicki seed + seeded Born sempling.

Tehnike:
  • 2-qubit unitarne matrice σ_k = A·I + A^{-1}·e_k,  σ_k^{-1} = A^{-1}·I + A·e_k.
  • Pleteno slovo w(j_target) od 10 generatora (forward k=0..4, backward k=4..0),
    sa ε_k = −1 ako bit_k(j_target) = 1, inače ε_k = +1.
  • U_b = ρ(w) — sekvencijalna primena 2-qubit gejtova.
  • P(j) = |⟨j|U_b|ψ_init⟩|², Born sempling iz uslovne (valid-masked) distribucije.
  • Dijagnostika: pseudo-Markov trace τ_J = ⟨0|U_b|0⟩ i fidelity F(j_target).
"""
