## Setup Instructions
`uv python pin 3.11`


## Proposed structure
```bash
issarkfcm/
├── data/                  # BrainWeb / IBSR downloads, synthetic images
├── src/
│   ├── fcm.py             # standard FCM (Algorithm 1)
│   ├── kfcm.py            # kernel FCM
│   ├── arkfcm.py          # adaptive regularized KFCM
│   ├── ghm_wavelet.py     # custom multiwavelet decomposition
│   ├── quadratic_dist.py  # quadratic polynomial distance
│   ├── cnn.py             # LeNet-style CNN (Fig. 5)
│   ├── de.py              # Differential Evolution (DE) loop
│   ├── issarkfcm.py       # final combined algorithm
│   └── metrics.py         # Vpc, Vpe, Vxb
├── experiments/           # scripts reproducing Tables 2-3, Figs 18-20
└── requirements.txt
```