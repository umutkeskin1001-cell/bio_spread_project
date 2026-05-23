import torch

from dna_sentinel import features
from dna_sentinel.features import MultiScaleKmerConfig, MultiScaleKmerExtractor, preprocess_all_features
from dna_sentinel.utils import LabeledSequence


def test_extract_shapes():
    ext = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=256))
    feat, spec, mask, sids = ext.extract("ATGCGT" * 200)
    total = sum(MultiScaleKmerConfig.max_windows)
    assert feat.shape == (total, 256)
    assert spec.shape == (total, 512)
    assert mask.shape == (total,)
    assert sids.shape == (total,)
    assert mask.any()


def test_rc_consensus_makes_features_symmetric():
    from dna_sentinel.utils import revcomp
    ext = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=256, rc_consensus=True))
    seq = "ATGCGT" * 1000
    f1, s1, m1, _ = ext.extract(seq)
    f2, s2, m2, _ = ext.extract(revcomp(seq))
    assert torch.allclose(f1[m1].sum(dim=0), f2[m2].sum(dim=0), atol=1e-4)
    assert torch.allclose(s1[m1].sum(dim=0), s2[m2].sum(dim=0), atol=1e-4)


def test_spectral_extraction_bins_magnitude_and_phase(monkeypatch):
    calls = []
    original = features.log_bin_spectral

    def spy(values, low_res=64, n_bins=64):
        calls.append((low_res, n_bins))
        return original(values, low_res=low_res, n_bins=n_bins)

    monkeypatch.setattr(features, "log_bin_spectral", spy)

    ext = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=256, rc_consensus=False))
    _, spec, _, _ = ext.extract("ATGCGT" * 200)

    assert spec.shape == (sum(MultiScaleKmerConfig.max_windows), 512)
    assert calls == [(32, 32)] * 6


def test_extractor_uses_configured_stride_for_window_sampling():
    cfg = MultiScaleKmerConfig(
        window_sizes=(8,),
        strides=(4,),
        max_windows=(10,),
        n_features=256,
        rc_consensus=False,
    )
    ext = MultiScaleKmerExtractor(cfg)

    _, _, mask, _ = ext.extract("ACGT" * 5)

    assert int(mask.sum().item()) == 4


def test_extractor_includes_unaligned_tail_window():
    cfg = MultiScaleKmerConfig(
        window_sizes=(8,),
        strides=(5,),
        max_windows=(10,),
        n_features=256,
        rc_consensus=False,
    )
    ext = MultiScaleKmerExtractor(cfg)

    _, _, mask, _ = ext.extract("ACGT" * 5 + "AC")

    assert int(mask.sum().item()) == 4


def test_kmer_counts_ignore_padding_bases():
    cfg = MultiScaleKmerConfig(
        window_sizes=(8,),
        strides=(8,),
        max_windows=(1,),
        ngram_min=4,
        ngram_max=4,
        n_features=5376,
        rc_consensus=False,
    )
    ext = MultiScaleKmerExtractor(cfg)

    feat, _, mask, _ = ext.extract("CCCC")

    cccc_index = 85
    assert mask.tolist() == [True]
    assert torch.count_nonzero(feat[0]).item() == 1
    assert torch.isclose(feat[0, cccc_index], torch.tensor(1.0))


def test_spectral_features_do_not_encode_padding_as_a_bases():
    cfg = MultiScaleKmerConfig(
        window_sizes=(8,),
        strides=(8,),
        max_windows=(1,),
        n_features=256,
        rc_consensus=False,
    )
    ext = MultiScaleKmerExtractor(cfg)

    _, short_spec, _, _ = ext.extract("C")
    _, padded_spec, _, _ = ext.extract("CAAAAAAA")

    assert not torch.allclose(short_spec, padded_spec)


def test_length_weighting_preserves_window_coverage_signal():
    cfg = MultiScaleKmerConfig(
        window_sizes=(8,),
        strides=(8,),
        max_windows=(1,),
        ngram_min=4,
        ngram_max=4,
        n_features=5376,
        rc_consensus=False,
        length_weighting=True,
    )
    ext = MultiScaleKmerExtractor(cfg)

    feat, spec, _, _ = ext.extract("CCCC")

    assert torch.isclose(feat.norm(dim=1)[0], torch.tensor(0.5))
    assert torch.isclose(spec.norm(dim=1)[0], torch.tensor(0.5))


def test_coverage_feature_uses_last_spectral_dimension():
    cfg = MultiScaleKmerConfig(
        window_sizes=(8,),
        strides=(8,),
        max_windows=(1,),
        n_features=256,
        rc_consensus=False,
        coverage_feature=True,
    )
    ext = MultiScaleKmerExtractor(cfg)

    _, spec, _, _ = ext.extract("CCCC")

    assert torch.isclose(spec[0, -1], torch.tensor(0.5))


def test_parallel_preprocess_matches_serial(tmp_path):
    records = [
        LabeledSequence("s1", "ATGCGT" * 200, 1, 0, 1),
        LabeledSequence("s2", "GATTACA" * 180, 0, 1, 0),
    ]
    cfg = MultiScaleKmerConfig(n_features=256, rc_consensus=False)
    serial_path = tmp_path / "serial.pt"
    parallel_path = tmp_path / "parallel.pt"

    preprocess_all_features(records, cfg, serial_path, num_workers=1)
    preprocess_all_features(records, cfg, parallel_path, num_workers=2, parallel_threshold=1)

    serial = torch.load(serial_path, weights_only=True)
    parallel = torch.load(parallel_path, weights_only=True)

    for key in ["features", "spec_features", "masks", "scale_ids"]:
        assert torch.allclose(serial[key], parallel[key])
