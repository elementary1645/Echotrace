# CLAUDE.md — EchoTrace

Solo, one-day hackathon build (Hacktoberfest Hack Day Bengaluru, IEEE RIT; track: Healthcare, well-being & public services).
**Hard deadline: working live demo by 6:00 PM.** Simple and robust beats clever. Fewer files, working code.

## Working rules for Claude
- Work through the PHASES below **in order**. At the end of each phase: STOP, give exact **PowerShell** commands to verify it and say what the user should see. Do not start the next phase until the user says **"next"**.
- Claude cannot hear the mic: anything involving live audio ships with a script the user runs and reports back on.
- Platform: Windows 11 + PowerShell. Python = **3.12 venv at `backend/.venv`** (system Python is 3.14 — never use it for the project).
- All tunables live in `backend/echotrace/config.py`. No magic numbers elsewhere.
- Scripts in `backend/scripts/` are run from `backend/` with the venv active, e.g. `python scripts/list_devices.py`.

## Progress log
- [x] PHASE 0 — scaffold, requirements, .gitignore, LICENSE, CLAUDE.md, download_weights.py, list_devices.py (venv built, weights downloaded, committed)
- [ ] PHASE 1 — Hearing (PANNs live)
- [ ] PHASE 2 — Connecting (core logic)
- [ ] PHASE 3 — Serving
- [ ] PHASE 4 — Dashboard
- [ ] PHASE 5 — Proof + fallback
- [ ] PHASE 6 — Open-source polish

## Decisions / findings (update as we go)
- torch is installed separately from requirements.txt (CUDA wheel `cu128` for the RTX 4050; CPU wheel as fallback).
- panns_inference checks `os.path.getsize(weights) >= 3e8`, else it shells out to wget. `download_weights.py` must run first.
  Zenodo drops connections / 504s; the script resumes via HTTP Range (verified 206 support).
- Installed: torch 2.11.0+cu128 (CUDA OK on RTX 4050), numpy 2.5.3, librosa 1.0.0, panns_inference 0.1.1, torchlibrosa 0.1.0.
  CNN14 inference on a 1 s window: ~10 ms CUDA, ~46 ms CPU (both fine for a 0.5 s hop).
- **Mic (laptop Intel SST array) probe, 2026-10-08** (speaker noise burst, 48k x 2ch):
  - WASAPI #17: Windows voice processing (AEC/noise suppression) -> channels identical (corr 0.9996, lag 0), burst suppressed. Useless for localization.
  - MME #1 / DirectSound #8: flat -96.7 dBFS on all channels (dead).
  - **WDM-KS #20 "Microphone Array 2" (and #19 "Microphone Array 1")**: raw true stereo (corr ~0.73, GCC lag -1..-4 for a centred speaker). Chosen default.
  - WDM-KS is exclusive-access, and device indices shift when the Bluetooth headset (WH-XB910N) connects -> select devices by NAME substring, not index.
  - Laptop array mic spacing is probably ~5-8 cm, not 0.20 m -> measure it; max lag shrinks to ~7-11 samples, so L/C/R margins are tight.

---

# ORIGINAL BRIEF (verbatim)

I'm building "EchoTrace" SOLO, from scratch, in one day at a hackathon (Hacktoberfest Hack Day Bengaluru, IEEE RIT; track: Healthcare, well-being & public services). Hard deadline: a working live demo by 6:00 PM. Read this whole brief first. Save it as CLAUDE.md so the context survives across sessions. Then work through the phases below IN ORDER. At the end of each phase: stop, give me the exact PowerShell commands to verify it, and tell me what I should see. Don't start the next phase until I say "next". You cannot hear my microphone, so anything involving live audio must come with a script that I run and report back on.

## WHAT ECHOTRACE IS
Open-weight, fully local AI that turns environmental sound into situational awareness for emergency responders, firefighters, industrial-safety teams and smart-building operators, for when cameras can't see (smoke, darkness, blind spots). Existing audio AI labels sounds one at a time. EchoTrace:
1. DETECTS acoustic events with PANNs (CNN14, pretrained on AudioSet, 527 classes)
2. LOCALIZES each event approximately to LEFT / CENTRE / RIGHT using GCC-PHAT + TDOA on a 2-channel mic input (signal processing, not ML)
3. CORRELATES events across time and space into sequences (temporal event graph)
4. RAISES explainable GREEN / AMBER / RED alerts with deterministic plain-language summaries

It complements CCTV; it never replaces it. Tagline: "PANNs hears. EchoTrace connects."

DEMO TARGET: I play a repeatable 4-sound sequence into a live mic:
  FOOTSTEPS (LEFT) -> FOOTSTEPS (CENTRE) -> IMPACT (RIGHT) -> DISTRESS (RIGHT)
The dashboard labels each sound with time + confidence, places it L/C/R, links them into ONE sequence ("Possible correlated acoustic sequence, moving towards RIGHT") and explains why it was flagged.

METRICS I promised the judges, measured against a CLASSIFICATION-ONLY BASELINE (one that alerts whenever any single concerning label crosses the threshold):
  sequence-recognition accuracy, false-alert reduction, L/C/R accuracy, end-to-end latency.

## HARD CONSTRAINTS
- No proprietary/cloud AI API and no generative LLM anywhere. Summaries are template-based and deterministic.
- Model: PANNs CNN14 via the `panns_inference` package (code MIT; weights CC BY 4.0 from Zenodo, which must be attributed in the README). No training or fine-tuning.
- Stack: Python 3.12 venv (my system Python is 3.14; do NOT use it, because torch/librosa/numba may break), PyTorch, FastAPI, uvicorn, WebSockets, sounddevice, numpy, scipy. Frontend: React + Vite + TypeScript, plain CSS, no heavy UI library. Optional SQLite only if time allows.
- Platform: Windows 11 + PowerShell. Laptop has an RTX 4050, but everything must work on CPU too (auto-detect CUDA, allow --cpu).
- There is NO `wget` on this machine. panns_inference shells out to wget to fetch `Cnn14_mAP=0.431.pth` and `class_labels_indices.csv` into ~/panns_data/. Write a Python download script that places both files there BEFORE panns_inference is imported, is idempotent, and checks the file sizes (an empty file means the download failed).
- Privacy: local inference only; store event metadata; raw audio is saved only when I explicitly record a test clip.
- Ethics wording everywhere (UI + summaries): RED = "potentially significant sequence", never "confirmed emergency". No identity tracking, only a "possible acoustic trajectory". Every event shows its confidence; every alert shows why.
- SOURCE HONESTY: every message and every UI screen carries a source mode: LIVE | RECORDED | SIMULATED. Simulated or recorded data must never be shown as live (always-visible badge; SIMULATED gets a distinct warning colour).
- Modular pipeline (tagger, localizer, correlator are swappable behind small interfaces). MIT licence for my repo. Everything configurable from one config.py; no magic numbers scattered around.
- Keep it simple and robust over clever. I'm solo with one day, so prefer fewer files and working code over abstraction.

## ARCHITECTURE
```
echotrace/
  backend/
    echotrace/
      config.py               # sample rates, window/hop, thresholds, mic spacing, category map, pattern settings
      schemas.py              # Pydantic message models (the WS contract below)
      audio/sources.py        # LiveMicSource (sounddevice, 48 kHz, 2 ch, ONE device), WavReplaySource (real-time paced, stereo or mono WAV), SimSource (scripted synthetic events, no model needed)
      tagging/panns_tagger.py # 48k stereo -> downmix -> resample_poly to 32k mono -> rolling window (default 1.0 s, configurable to 2.0) with 0.5 s hop -> clipwise probs
      tagging/categories.py   # AudioSet label NAMES -> categories; resolve indices by name at startup and fail loudly if a name is missing
      detection/debounce.py   # per-category hysteresis (on ~0.30, off ~0.15, N-of-M windows) -> discrete events with t_start/t_end/peak confidence
      localization/gcc_phat.py  # GCC-PHAT on the raw 48k stereo, gated to the highest-energy frames near the event onset, band-limited, parabolic peak interpolation; tau -> angle = arcsin(clip(c*tau/d)); majority vote over frames
      localization/zones.py   # angle -> LEFT/CENTRE/RIGHT (CENTRE = |angle| < ~20 deg), sign flip from calibration, UNKNOWN if mono / low confidence
      correlation/graph.py    # events are nodes; link each new event to recent events within ~8 s; track zone progression as a "possible acoustic trajectory"
      correlation/patterns.py # ordered templates with max gaps, e.g.
                              #   MOVEMENT_IMPACT_DISTRESS: FOOTSTEPS+ -> IMPACT -> DISTRESS  => RED
                              #   IMPACT_DISTRESS: IMPACT -> DISTRESS => AMBER/RED
                              #   ALARM_EVACUATION: ALARM -> FOOTSTEPS (several)  => AMBER
                              #   GLASS_INTRUSION: GLASS -> FOOTSTEPS => AMBER
                              # single isolated events => GREEN (logged, no alert); that is the false-alert reduction vs baseline
      risk/engine.py          # current risk state + decay back to GREEN after quiet time
      risk/summary.py         # deterministic summary text, e.g. "14:02:11 - Footsteps heard moving LEFT -> CENTRE, followed 1.2 s later by an impact on the RIGHT (71%) and a possible distress vocalisation on the RIGHT (54%). Pattern: movement -> impact -> distress. Status RED: potentially significant sequence - verify."
      pipeline.py             # audio callback -> queue -> inference worker thread -> debounce -> localize -> correlate -> risk -> asyncio broadcast (loop.call_soon_threadsafe)
      baseline.py             # classification-only baseline alerter, used for metrics
      api/server.py           # FastAPI: WS /ws; GET /api/status; GET/POST /api/mode {source, device?, file?}; GET /api/events; POST /api/reset; serves the built frontend in production
    scripts/
      download_weights.py, list_devices.py, live_topk.py, record_clip.py, calibrate_direction.py, eval.py, ws_print.py
    tests/
  frontend/  (React + Vite + TS)
  recordings/  (demo clips + labels/*.json ground truth)
  docs/
  README.md, LICENSE (MIT), .gitignore (venv, node_modules, weights, large wavs)
```

CATEGORIES (map by AudioSet display name; verify every name exists in class_labels_indices.csv):
- FOOTSTEPS: "Walk, footsteps", "Run"
- IMPACT:    "Thump, thud", "Slam", "Bang", "Smash, crash", "Knock"
- DISTRESS:  "Screaming", "Yell", "Crying, sobbing", "Shout", "Groan", "Whimper"
- ALARM:     "Smoke detector, smoke alarm", "Fire alarm", "Alarm", "Siren", "Buzzer"
- GLASS:     "Glass", "Shatter", "Breaking"
- DOOR:      "Door", "Sliding door", "Cupboard open or close"

Category score = max over its member labels. These are starting points; I'll tune them from the live_topk output.

## WEBSOCKET CONTRACT (Pydantic in schemas.py + matching TS types in frontend/src/types.ts)
All messages: {type, data}. Every data object has `source` ("LIVE"|"RECORDED"|"SIMULATED"), `t` (monotonic seconds since session start) and `wall` (ISO time).
- frame:    {top:[{label,p}] (top 5), categories:{FOOTSTEPS:p,...}, rms_db:[l,r]}       (every hop; UI meters/debug)
- event:    {id, t_start, t_end, category, label, confidence, zone:"LEFT"|"CENTRE"|"RIGHT"|"UNKNOWN", angle_deg|null}
- sequence: {id, event_ids, pattern, risk:"GREEN"|"AMBER"|"RED", explanation:[str], summary, trajectory:[zone,...]}
- status:   {risk, latency_ms, device, sample_rate, channels, localization:"ON"|"OFF", model_device:"cpu"|"cuda"}

latency_ms = time from capture of the audio block that triggered the event -> WS send time.

## PHASES (solo critical path: the safest demo comes first)
PHASE 0 - Setup (keep it short)
  Scaffold the repo, requirements.txt, .gitignore, git init, MIT LICENSE, CLAUDE.md.
  Give me the PowerShell commands to create a Python 3.12 venv (`py -3.12 -m venv .venv`) and install everything (torch CPU or CUDA wheel, whichever fits; tell me which).
  scripts/download_weights.py (wget-free) and scripts/list_devices.py (input devices, channel counts, default sample rates; mark the devices that support 2 input channels).

PHASE 1 - Hearing (PANNs live)
  tagging/ + LiveMicSource + scripts/live_topk.py (--device --window --hop --cpu): prints the top-5 labels, the category scores and the inference ms every hop, with a live RMS meter per channel.
  scripts/record_clip.py: record N seconds of 48k stereo from a chosen device to recordings/.
  Tests: category names resolve; resampling shape and rate are correct; tagger smoke test on a generated WAV.
  -> I'll run live_topk.py while making each demo sound and report which classes fire reliably, so we can tune categories and thresholds.

PHASE 2 - Connecting (core logic, fully unit-testable without a mic)
  debounce.py, gcc_phat.py + zones.py, graph.py, patterns.py, risk engine, summary.py, baseline.py, pipeline.py wired to all three sources.
  SimSource emits the scripted demo sequence (plus a couple of distractor single events) with zones, clearly tagged SIMULATED.
  Tests: hysteresis merges/splits correctly; GCC-PHAT recovers known delays from synthetic signals (white noise and a click delayed by +/-k samples), giving the correct L/C/R; the demo sequence -> one RED sequence with a correct explanation; isolated single events -> GREEN (no alert) while the baseline DOES alert; summary text is deterministic.
  If the device has only 1 channel or the channels are identical (correlation ~1.0 at zero lag), localization switches OFF automatically, zones become UNKNOWN and the status reports it. Classification + correlation must keep working (this is the graceful fallback).
  scripts/calibrate_direction.py: "clap on the LEFT, now the RIGHT"; reports the measured angles, saves the sign/offset to a local calibration JSON.

PHASE 3 - Serving
  FastAPI server + WS broadcast + REST endpoints + source switching at runtime (LIVE / RECORDED wav / SIMULATED), scripts/ws_print.py test client. One command starts everything.

PHASE 4 - Dashboard (React + Vite + TS, dark "ops console" look, readable on a projector)
  - StatusBanner: big GREEN/AMBER/RED with cautious wording + an always-visible SOURCE badge + latency + localization ON/OFF.
  - AcousticMap: top-down room view with the mic pair at the bottom and LEFT/CENTRE/RIGHT zones; events appear as dots in their zone, coloured by category, fading over ~15 s; arrows connect events in the active sequence (the possible acoustic trajectory).
  - Timeline: horizontal lanes per category, last 60 s, event bars with confidence; sequence links drawn between them.
  - DetailPanel: click a sequence -> pattern, risk, summary, the "why" explanation list, every contributing event with time/confidence/zone.
  - Live meters: per-channel level + current top-5 labels (proves it's really listening).
  - Controls: source switcher (Live / Replay file / Simulation), reset, auto-reconnecting WS.
  - MetricsPanel: shows the latest eval.py results (loaded from a JSON the backend serves).
  Vite proxy to the backend in dev; the backend serves the build for the demo.

PHASE 5 - Proof + fallback
  recordings/labels/*.json ground-truth format (expected events, zones, expected pattern) and scripts/eval.py: replays each labelled WAV through the SAME pipeline (as fast as possible, not real time), and computes:
    sequence-recognition accuracy, false alerts EchoTrace vs baseline (and % reduction), L/C/R accuracy, mean/p95 latency (latency from a separate live/real-time run).
  Writes docs/metrics.json + a markdown table. Includes negative clips (lone door slam, lone footsteps, background chatter) so the baseline's false alerts show up.
  Fallback ladder for the demo, switchable in one click: LIVE -> RECORDED demo clip -> SIMULATED (labelled). Classification-only keeps the timeline running even if localization fails.

PHASE 6 - Open-source polish
  README: problem, how it works (pipeline diagram in mermaid), setup on Windows, how to run the demo, how to record + evaluate, metrics table, ethics & limitations (zones only, front/back ambiguity, noisy rooms, not a replacement for responders), attribution (PANNs paper Kong et al. 2020, qiuqiangkong/audioset_tagging_cnn MIT, panns_inference, Zenodo weights CC BY 4.0, AudioSet). Clean up and make sure a fresh clone runs from the README.

## KNOWN TECHNICAL GOTCHAS (design around these)
- GCC-PHAT only works if both channels come from ONE device with ONE clock (stereo USB mic / 2-input USB interface / true stereo laptop array). Two separate USB mics drift, so don't support that path; detect identical channels and disable localization.
- With mic spacing d (configurable, default 0.20 m), max delay = d/343 s (~28 samples at 48 kHz). Search only lags within +/-max_lag, interpolate the peak, and use only onset/high-energy frames, because reverb in a hall otherwise dominates.
- CNN14 was trained on 10 s clips, so short windows give lower probabilities. Keep window/hop/thresholds configurable and expect to tune them in Phase 1.
- Run inference in a worker thread, never in the sounddevice callback or the asyncio loop. Warm up the model once at startup. Drop stale windows if inference falls behind, and report it.
- Windows "audio enhancements" / noise suppression can merge or alter channels. Mention this in the README troubleshooting section.
