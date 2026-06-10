# Antipodal Complementarity in a 3×3 Topological Partition of China's Climate

Verification and analysis code accompanying the paper:

**Antipodal Complementarity in a 3×3 Topological Partition of China's Climate: A Three-Contribution Decomposition**

Author: Jinguo Ling (凌进国) — ORCID: [0009-0009-1406-4551](https://orcid.org/0009-0009-1406-4551)

## Repository Structure

```
├── scripts/
│   ├── luoshu_character_test.py       # L1: Luoshu character test (antipodal vs non-antipodal, 105 exhaustive pairings)
│   ├── opposite_palace_test.py        # Opposite-palace pairing significance test
│   ├── ew_complementarity_analysis.py # Contribution 3: E–W asymmetry deep-dive (monsoon vs elevation)
│   ├── spacetime_coupling_test.py     # L2: Space–time coupling test (true negative confirmation)
│   ├── seasonal_quantization_test.py  # Z₅ five-phase effect independent test
│   ├── axis_specific_experiments.py   # Axis-specific rotation experiments
│   ├── fwer_correction.py            # Family-wise error rate correction for multiple tests
│   ├── expA_prate_dswrf.py           # Supplementary: precipitation & downward SW radiation
│   ├── expB_directional.py           # Supplementary: directional variable tests
│   ├── expC_disaster_palace.py       # Supplementary: disaster palace analysis
│   ├── expD_xialin_gong.py           # Supplementary: Xia-Lin palace remapping
│   └── expE_permutation.py           # Supplementary: permutation-based robustness
├── data/
│   ├── exhaustive_pairings.csv        # All 105 pairing schemes with CV, antipodal count, and rank
│   └── sliding_windows.csv            # 18 sliding 30-year windows (1901–1988) with per-window statistics
├── figures/                           # Paper figures (PNG + PDF, 300 DPI, Matplotlib)
│   ├── Figure1_洛书框架.png/.pdf       # 3×3 partition framework
│   ├── Figure2_三贡献分解.png/.pdf     # Three-contribution decomposition
│   ├── Figure3_零模型.png/.pdf         # Null model hierarchy
│   └── Figure4_时间稳健性.png/.pdf     # Temporal robustness (18 windows)
├── requirements.txt
├── LICENSE
└── README.md
```

## Key Results Reproduced

| Result | Script | Output |
|--------|--------|--------|
| Luoshu ranks 1/105 in exhaustive pairing | `luoshu_character_test.py` | CV = 0.061, rank 1 |
| Three-tiered null model hierarchy | `luoshu_character_test.py` | p ≤ 0.010 all tiers |
| E–W asymmetry (3↔7): 2.0×, p < 0.001 | `ew_complementarity_analysis.py` | Monsoon amplification |
| 18-window temporal robustness | `data/sliding_windows.csv` | meta-CV = 0.030 |
| Latitude gradient: R² = 0.917 | `luoshu_character_test.py` | Contribution 1 |
| Z₅ five-phase effect | `seasonal_quantization_test.py` | Independent validation |

## Data Requirements

The scripts require the following publicly available datasets:

| Dataset | Description | Source |
|---------|-------------|--------|
| UDel v501 | 0.5° gridded monthly surface air temperature, 1900–2017 | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.UDel_AirT_Precip.html) |
| NCEP/NCAR Reanalysis | Monthly mean surface variables | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.surface.html) |
| CUG-CMA 2.5° | Chinese station gridded temperature dataset | China Meteorological Administration |

Place downloaded data files in `data/` directory. Pre-computed results (exhaustive pairings, sliding windows) are included and do not require raw data to inspect.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run core analysis
python scripts/luoshu_character_test.py
python scripts/ew_complementarity_analysis.py

# Run supplementary experiments
python scripts/expA_prate_dswrf.py
python scripts/expB_directional.py
```

## Citation

If you use this code, please cite:

> Ling, J. (2026). Antipodal Complementarity in a 3×3 Topological Partition of China's Climate: A Three-Contribution Decomposition. *Proceedings of the National Academy of Sciences* (submitted). DOI: 10.5281/zenodo.20477230

## License

MIT License. See [LICENSE](LICENSE) for details.
