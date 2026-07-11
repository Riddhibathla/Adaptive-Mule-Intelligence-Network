# Adaptive Mule Intelligence Network & Satark AI 🛡️

An AI-powered fraud intelligence prototype for detecting mule-account activity, suspicious transactions, and consumer identity misuse across Indian digital payment channels.

## ✨ What It Does

- 🧠 **Adaptive Mule Intelligence Network**: B2G/B2B dashboard for mule-risk scoring, fraud alerts, graph intelligence, feedback learning, and compliance workflows.
- 👥 **Satark AI**: B2C self-check page where citizens can check whether their **PAN or phone number** may be linked to suspicious account-opening activity.
- 🎙️ **Regional voice assistance**: Supports multilingual guidance and spoken responses for Hindi, Marathi, Tamil, Telugu, Punjabi, Kannada, Bengali, and English.
- 🔐 **Privacy-first checks**: Results are masked and designed for safe consumer-facing use.

## 🗂️ Folder Architecture

```text
Adaptive Mule Intelligence Network/
├─ backend/                         # Python backend API
│  ├─ server.py                     # App entry point
│  ├─ config.py                     # Paths, host, port, config
│  ├─ controllers/                  # API and ADAPT workflow handlers
│  │  ├─ api_controller.py
│  │  └─ adapt_controller.py
│  ├─ models/
│  │  └─ risk_model.py              # Mule-risk, identity misuse, graph logic
│  ├─ routes/
│  │  └─ router.py                  # GET/POST route mapping
│  └─ server/
│     ├─ app.py                     # HTTP server startup
│     └─ http_handler.py            # Static frontend + JSON API serving
│
├─ frontend/                        # User interfaces
│  ├─ index.html                    # Adaptive Mule Intelligence dashboard
│  ├─ app.js                        # Dashboard logic
│  ├─ selfcheck.html                # Satark AI consumer page
│  ├─ selfcheck.js                  # Self-check, language, voice logic
│  └─ styles.css                    # Shared + Satark AI styling
│
├─ data/                            # Local training/demo datasets
│  ├─ adaptive_mule_training_dataset.csv
│  ├─ adapt_instruction_dataset.csv
│  └─ kaggle/
│     ├─ creditcard.csv
│     └─ PS_20174392719_1491204439457_log.csv
│
└─ README.md
```

## 🚀 Run Locally

```powershell
python backend\server.py
```

Open:

- 🖥️ Dashboard: `http://127.0.0.1:5173`
- 📱 Satark AI: `http://127.0.0.1:5173/selfcheck.html`

No package installation is required; the backend uses Python standard library modules only.

## 🧩 Key APIs

- `GET /api/model` - model metadata and validation profile
- `GET /api/state` - live dashboard state
- `POST /api/transactions` - ingest a transaction
- `POST /api/identity-checks` - run Satark AI PAN/phone misuse check

## 🎯 Vision

The platform combines compliance-grade mule detection with citizen-friendly identity protection, helping institutions and consumers respond faster to suspicious financial activity.
