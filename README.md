# 🧬 GENE — AI Text Generation Agent

> **UI note:** `app.py` now renders the GENE dashboard theme — sidebar
> settings panel, live word/character/token/runtime stat cards, chat-style
> transcript, History/Saved Prompts/Settings/About views, and a dark/light
> toggle. `agent.py` is unchanged: every response still comes from a real
> Hugging Face `text-generation` pipeline call, nothing is simulated.

A Streamlit application that lets users generate stories, emails, summaries,
explanations, or creative text from a single prompt, powered by
**Qwen/Qwen2.5-0.5B-Instruct** via Hugging Face Transformers.

## Architecture

```
User → Streamlit UI → AI Agent Controller → HF Transformers Pipeline
     → Qwen2.5-0.5B-Instruct → Generated Text → Display Output
```

- **`app.py`** — Streamlit UI: prompt input, task selection, generation
  parameter controls, output display and history.
- **`agent.py`** — AI Agent Controller: receives the prompt, builds a
  task-specific instruction, validates input, and manages the generation
  call to the Hugging Face pipeline.

## Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`. The model downloads automatically
the first time you click **Generate** (cached afterward by
`st.cache_resource` and Hugging Face's local cache).

> **Note:** First model load can take a minute or two depending on your
> connection. GPU is not required — the 0.5B model runs fine on CPU, though
> a GPU (CUDA) speeds things up if `torch` detects one automatically.

## Project structure

```
gene_agent/
├── app.py                  # Streamlit UI
├── agent.py                # AI Agent Controller
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme & server config
└── README.md
```

## Deployment

### Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set the main file to `app.py`.
3. Deploy. (Free tier CPU is sufficient for the 0.5B model, though first
   load will be slow.)

### Docker
```bash
docker build -t gene-agent .
docker run -p 8501:8501 gene-agent
```
(See `Dockerfile` included in this project.)

### Hugging Face Spaces
1. Create a new Space with SDK = Streamlit.
2. Upload all files in this folder (rename nothing).
3. Space builds automatically from `requirements.txt` and runs `app.py`.

## Customization

- **Change the model:** edit the "Model name" field in the sidebar, or
  change `DEFAULT_MODEL_NAME` in `agent.py` to any other Hugging Face
  `text-generation`-compatible chat model (e.g. a larger Qwen or Llama
  variant), provided you have the hardware for it.
- **Add a new task type:** add an entry to `TASK_INSTRUCTIONS` in
  `agent.py` — it will automatically appear in the UI's task dropdown.
- **Tune generation defaults:** adjust `GenerationParams` defaults in
  `agent.py`, or just use the sidebar sliders at runtime.

## License

MIT — do whatever you like with it.
