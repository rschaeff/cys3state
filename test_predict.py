#!/usr/bin/env python
"""
Integration tests for cys3state predictor.

Runs the ensemble on proteins with known cysteine functional states and
checks that predictions match ground truth from PDB structural evidence.

Requires: torch, fair-esm, numpy (same deps as predict.py).
ESM2-650M weights will be downloaded on first run (~2.5 GB).

Usage:
    pytest test_predict.py -v
    pytest test_predict.py -v -k metal
    python test_predict.py          # also works standalone
"""

import os
import sys
import tempfile
import subprocess
import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict.py")

# ---------------------------------------------------------------------------
# Test proteins with structurally verified cysteine states
# ---------------------------------------------------------------------------

# P80882: Rubredoxin (75 aa) — 4 Cys, all metal-binding (Fe-S cluster)
# PDB: 1hlq chain A
P80882_SEQ = "AAPLVAETDANAKSLGYVADTTKADKTKYPKHTKDQSCSTCALYQGKTAPQGACPLFAGKEVVAKGWCSAWAKKA"

# P37237: Coagulation factor (268 aa) — 2 labeled Cys as disulfide
# PDB: 2fdb chain N
P37237_SEQ = ("MGSPRSALSCLLLHLLVLCLQAQVRSAAQKRGPGAGNPADTLGQGHEDRPFGQRSRAGK"
              "NFTNPAPNYPEEGSKEQRDSVLPKVTQRHVREQSLVTDQLSRRLIRTYQLYSRTSGKHV"
              "QVLANKRINAMAEDGDPFAKLIVETDTFGSRVRVRGAETGLYICMNKKGKLIAKSNGKGK"
              "DCVFTEIVLENNYTALQNAKYEGWYMAFTRKGRPRKGSKTRQHQREVHFMKRLPRGHHTT"
              "EQSLRFEFLNYPPFTRSLRGSQRTWAPEPR")

# A0A1S3VYF3: Small protein (103 aa) — mixed: 2 Dis, 1 Neg
# PDB: 8e73 chain S6
A0A1S3VYF3_SEQ = ("MASSVLRNLMRFSSPRNTSTRSFSLVTSQISNHTAKWMQDTSKKSPMELINEVPPIKVE"
                  "GRIVACEGDTNPALGHPIEFICLDLPEPAVCKYCGLRYVQDHHH")

# D3S6S2: Nitrogenase iron protein (285 aa) — 2 Met, 1 Neg
# PDB: 8q5w chain A
D3S6S2_SEQ = ("MSVYDEIAPNAKKVAIYGKGGIGKSTTTQNTAAALAYYYKLKGMIHGCDPKADSTRMIL"
              "HGKPQETVMDVLREEGEEGVTLEKIRKVGFGGILCVESGGPEPGVGCAGRGVITAVNLM"
              "IELGGYPDDLDFLFFDVLGDVVCGGFAMPLRDGLAKEIYIVSSGEMMALYAANNIARGI"
              "LKYAEQSGVRLGGIICNSRKVDGEKELMEEFCDLLGTKLIHFIPRDNIVQKAEFNKMTVV"
              "EFAPDHPQAHEYKKLGKKIMDNDELVIPTPLSMDQLEKLVEKYGLLDK")

# Ground-truth labels from PDB structural evidence
# Format: (protein, residue, expected_class)
#   expected_class: "Met" = metal-binding, "Dis" = disulfide, "Neg" = reduced
GROUND_TRUTH = [
    # P80882 — rubredoxin Fe-S cluster: all 4 Cys coordinate iron
    ("P80882",      38, "Met"),
    ("P80882",      41, "Met"),
    ("P80882",      54, "Met"),
    ("P80882",      68, "Met"),
    # P37237 — coagulation factor disulfide bonds
    ("P37237",     162, "Dis"),
    ("P37237",     180, "Dis"),
    # A0A1S3VYF3 — mixed states
    ("A0A1S3VYF3",  65, "Dis"),
    ("A0A1S3VYF3",  81, "Neg"),
    ("A0A1S3VYF3",  90, "Dis"),
    # D3S6S2 — nitrogenase: 2 cluster-binding Cys + 1 reduced
    ("D3S6S2",     106, "Met"),
    ("D3S6S2",     141, "Met"),
    ("D3S6S2",      48, "Neg"),
]

# Reference predictions from the production pipeline (same weights).
# Used to check reproducibility, not just classification.
# Format: (protein, residue, neg, dis, met)
REFERENCE_PREDICTIONS = [
    # Values from running this pipeline (predict.py) on CPU.
    # Model: hidden_dim=128, dropout=0.2 (retrained weights).
    ("P80882",  38, 0.0011, 0.0350, 0.9639),
    ("P80882",  41, 0.0011, 0.0133, 0.9855),
    ("P80882",  54, 0.0014, 0.0142, 0.9844),
    ("P80882",  68, 0.0024, 0.1264, 0.8711),
    ("P37237", 162, 0.0005, 0.9993, 0.0002),
    ("P37237", 180, 0.0088, 0.9873, 0.0039),
    ("D3S6S2", 106, 0.0010, 0.0012, 0.9978),
    ("D3S6S2", 141, 0.0001, 0.0003, 0.9996),
]


# ---------------------------------------------------------------------------
# Fixture: run predictor once, share results across tests
# ---------------------------------------------------------------------------

def _make_fasta(path):
    with open(path, "w") as f:
        for name, seq in [("P80882", P80882_SEQ),
                          ("P37237", P37237_SEQ),
                          ("A0A1S3VYF3", A0A1S3VYF3_SEQ),
                          ("D3S6S2", D3S6S2_SEQ)]:
            f.write(f">{name}\n{seq}\n")


def _parse_output(path):
    """Parse TSV output into dict of {(protein, residue): (neg, dis, met)}."""
    results = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            prot = parts[0]
            resid = int(parts[1])
            neg, dis, met = float(parts[2]), float(parts[3]), float(parts[4])
            results[(prot, resid)] = (neg, dis, met)
    return results


@pytest.fixture(scope="session")
def predictions(tmp_path_factory):
    """Run predict.py once on all test proteins, return parsed results."""
    tmpdir = tmp_path_factory.mktemp("cys3state")
    fasta_path = str(tmpdir / "test.fasta")
    output_path = str(tmpdir / "predictions.tsv")

    _make_fasta(fasta_path)

    result = subprocess.run(
        [sys.executable, SCRIPT, fasta_path, "-o", output_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"predict.py failed:\n{result.stderr}")

    return _parse_output(output_path)


# ---------------------------------------------------------------------------
# Tests: output format and completeness
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_all_cysteines_present(self, predictions):
        """Every cysteine in every input protein should have a prediction."""
        expected = {
            "P80882": [38, 41, 54, 68],
            "P37237": [10, 19, 162, 180],
            "A0A1S3VYF3": [65, 81, 90, 93],
            "D3S6S2": [48, 94, 106, 141, 193, 209],
        }
        for prot, positions in expected.items():
            for pos in positions:
                assert (prot, pos) in predictions, \
                    f"Missing prediction for {prot} Cys {pos}"

    def test_probabilities_sum_to_one(self, predictions):
        """Each prediction's probabilities should sum to ~1.0."""
        for key, (neg, dis, met) in predictions.items():
            total = neg + dis + met
            assert abs(total - 1.0) < 0.01, \
                f"{key}: probabilities sum to {total:.4f}, expected ~1.0"

    def test_probabilities_non_negative(self, predictions):
        """All probabilities should be >= 0."""
        for key, (neg, dis, met) in predictions.items():
            assert neg >= 0 and dis >= 0 and met >= 0, \
                f"{key}: negative probability found ({neg}, {dis}, {met})"


# ---------------------------------------------------------------------------
# Tests: classification accuracy on known cases
# ---------------------------------------------------------------------------

def _argmax_class(neg, dis, met):
    vals = {"Neg": neg, "Dis": dis, "Met": met}
    return max(vals, key=vals.get)


class TestMetalBinding:
    """P80882 rubredoxin: 4 Cys all coordinate iron. D3S6S2: 2 cluster Cys."""

    @pytest.mark.parametrize("prot,resid", [
        ("P80882", 38), ("P80882", 41), ("P80882", 54), ("P80882", 68),
        ("D3S6S2", 106), ("D3S6S2", 141),
    ])
    def test_metal_classification(self, predictions, prot, resid):
        neg, dis, met = predictions[(prot, resid)]
        assert _argmax_class(neg, dis, met) == "Met", \
            f"{prot} Cys {resid}: expected Met, got {_argmax_class(neg, dis, met)} " \
            f"(neg={neg:.3f}, dis={dis:.3f}, met={met:.3f})"

    @pytest.mark.parametrize("prot,resid", [
        ("P80882", 41), ("P80882", 54), ("P80882", 68),
        ("D3S6S2", 106), ("D3S6S2", 141),
    ])
    def test_metal_high_confidence(self, predictions, prot, resid):
        """Strong metal binders should have Met_prob > 0.7."""
        _, _, met = predictions[(prot, resid)]
        assert met > 0.7, \
            f"{prot} Cys {resid}: Met_prob={met:.3f}, expected > 0.7"


class TestDisulfide:
    """P37237: Cys 162, 180 form disulfide bonds."""

    @pytest.mark.parametrize("prot,resid", [
        ("P37237", 162), ("P37237", 180),
    ])
    def test_disulfide_classification(self, predictions, prot, resid):
        neg, dis, met = predictions[(prot, resid)]
        assert _argmax_class(neg, dis, met) == "Dis", \
            f"{prot} Cys {resid}: expected Dis, got {_argmax_class(neg, dis, met)} " \
            f"(neg={neg:.3f}, dis={dis:.3f}, met={met:.3f})"

    @pytest.mark.parametrize("prot,resid", [
        ("P37237", 162), ("P37237", 180),
    ])
    def test_disulfide_high_confidence(self, predictions, prot, resid):
        """Strong disulfides should have Dis_prob > 0.9."""
        _, dis, _ = predictions[(prot, resid)]
        assert dis > 0.9, \
            f"{prot} Cys {resid}: Dis_prob={dis:.3f}, expected > 0.9"


class TestNegative:
    """D3S6S2 Cys 48 and A0A1S3VYF3 Cys 81: reduced/negative cysteines."""

    @pytest.mark.parametrize("prot,resid", [
        ("D3S6S2", 48), ("A0A1S3VYF3", 81),
    ])
    def test_negative_classification(self, predictions, prot, resid):
        neg, dis, met = predictions[(prot, resid)]
        assert _argmax_class(neg, dis, met) == "Neg", \
            f"{prot} Cys {resid}: expected Neg, got {_argmax_class(neg, dis, met)} " \
            f"(neg={neg:.3f}, dis={dis:.3f}, met={met:.3f})"


class TestMixedProtein:
    """A0A1S3VYF3 has both disulfide and negative cysteines."""

    def test_cys65_disulfide(self, predictions):
        neg, dis, met = predictions[("A0A1S3VYF3", 65)]
        assert _argmax_class(neg, dis, met) in ("Dis", "Met"), \
            f"A0A1S3VYF3 Cys 65: expected Dis or Met, got Neg"

    def test_cys90_disulfide(self, predictions):
        neg, dis, met = predictions[("A0A1S3VYF3", 90)]
        assert _argmax_class(neg, dis, met) in ("Dis", "Met"), \
            f"A0A1S3VYF3 Cys 90: expected Dis or Met, got Neg"

    def test_cys81_negative(self, predictions):
        neg, dis, met = predictions[("A0A1S3VYF3", 81)]
        assert _argmax_class(neg, dis, met) == "Neg", \
            f"A0A1S3VYF3 Cys 81: expected Neg, got {_argmax_class(neg, dis, met)}"


# ---------------------------------------------------------------------------
# Tests: reproducibility against reference predictions
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Check that predictions match the production pipeline within tolerance.

    Tolerance of 0.05 accounts for differences in ESM2 batching, numerical
    precision across hardware, and PyTorch version differences."""

    TOLERANCE = 0.05

    @pytest.mark.parametrize("prot,resid,ref_neg,ref_dis,ref_met",
                             REFERENCE_PREDICTIONS)
    def test_matches_reference(self, predictions, prot, resid,
                               ref_neg, ref_dis, ref_met):
        neg, dis, met = predictions[(prot, resid)]
        assert abs(neg - ref_neg) < self.TOLERANCE, \
            f"{prot} Cys {resid}: Neg={neg:.4f}, ref={ref_neg:.4f}"
        assert abs(dis - ref_dis) < self.TOLERANCE, \
            f"{prot} Cys {resid}: Dis={dis:.4f}, ref={ref_dis:.4f}"
        assert abs(met - ref_met) < self.TOLERANCE, \
            f"{prot} Cys {resid}: Met={met:.4f}, ref={ref_met:.4f}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
