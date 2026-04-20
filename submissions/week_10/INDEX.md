# Week 10 Submission Index
## Final Submission — PCA-Guided GP/EI

**Date**: March 30, 2026
**Status**: ✅ Ready for Portal Submission
**Strategy**: PCA principal directions of winner clusters + GP Expected Improvement
**Budget**: Final queries (W1–W10 complete)

---

## 📁 File Structure

```
week_10/
├── queries.py                    # PRIMARY SUBMISSION FILE ⭐
├── TECHNICAL_REPORT_WEEK10.md   # Full technical report + retrospective
├── WEEK9_REFLECTION.md          # W9 discussion board reflection
├── COMPLETION_SUMMARY.md        # (pre-existing W10 datasheet/model card summary)
├── DISCUSSION_POST_TRANSPARENCY.md
└── INDEX.md                     # This file
```

---

## 🎯 Portal Format (copy-paste)

```
F1: 0.101613-0.728058
F2: 0.501828-0.474203
F3: 0.483270-0.740795-0.558480
F4: 0.555318-0.550912-0.531991-0.670623
F5: 0.000000-1.000000-0.319539-0.210119
F6: 0.264033-0.460343-0.695113-0.856664-0.528116
F7: 0.228555-0.060712-0.586883-0.448725-1.000000-0.867749
F8: 0.019493-0.553542-0.280020-0.100114-0.630907-0.331912-0.281512-0.000000
```

---

## 📊 Week 9 Results (what informed W10)

| F | W8 | W9 | Change | Status |
|:---:|---:|---:|---:|:---|
| F1 | ≈0 | ≈0 | — | Noise floor |
| F2 | 0.033 | **0.481** | +1362% | Strong recovery |
| F3 | −0.138 | **−0.005** | +96% | 🏆 All-time best |
| F4 | −5.556 | **−4.635** | +17% | 🏆 All-time best |
| F5 | 1.149 | **77.553** | +6650% | Near W6 peak |
| F6 | −1.570 | −0.988 | +37% | Improved |
| F7 | 0.319 | **0.417** | +31% | 🏆 All-time best |
| F8 | 7.823 | **9.529** | +22% | 🏆 All-time best |
| **Portfolio** | **4.49** | **82.35** | **+3,899%** | 🚀 |

---

## 🔬 PCA Key Findings

| F | PC1 Variance | Key Dimensions |
|:---:|:---:|:---|
| F4 | 98.8% | Dims 1, 2, 4 dominate; dim 3 irrelevant |
| F5 | 98.3% | Low dim-2, high dim-4 = peak region |
| F8 | 81.0% | Dims 3, 5, 8 carry signal |

---

## 🎓 Project Summary (W1–W10)

**Best portfolio**: W9 = 82.35
**All-time bests achieved in W9**: F3, F4, F7, F8
**Defining lesson**: PCA over winner clusters identifies the 1–2 directions
that carry real signal. With N ≤ 21, searching the full D-dimensional space
is noise. Searching along principal components is signal.
