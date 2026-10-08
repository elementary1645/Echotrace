# EchoTrace

**PANNs hears. EchoTrace connects.**

Open-weight, fully local acoustic situational awareness for responders and building operators, for when cameras can't see.
EchoTrace detects sound events (PANNs CNN14), places them LEFT / CENTRE / RIGHT (GCC-PHAT on a stereo mic),
links them into sequences over time and space, and raises explainable GREEN / AMBER / RED alerts.
It complements CCTV and human responders; it never replaces them.

> Work in progress (hackathon build). Full README with setup, demo, metrics, ethics and attribution comes in the final phase.

## Quick setup (Windows, PowerShell)
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
pip install -r requirements.txt
python scripts/download_weights.py
python scripts/list_devices.py
```

## Attribution
- PANNs: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM TASLP 2020.
- Code: [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) (MIT), [panns_inference](https://github.com/qiuqiangkong/panns_inference) (MIT).
- CNN14 weights: [Zenodo record 3987831](https://zenodo.org/record/3987831), CC BY 4.0.
- Labels: Google AudioSet ontology (CC BY 4.0).

## License
MIT (this repository's code). Model weights are third-party, CC BY 4.0, downloaded separately.
