# GENE — Generative ENgine Agent

> GENE is a local AI writing assistant built with Streamlit,
> Hugging Face Transformers, and Qwen2.5-Instruct.

GENE allows users to generate stories, professional emails, summaries,
explanations, and creative text from a simple prompt.

The application combines a Streamlit interface with a task-aware AI
agent controller that manages prompt validation, instruction generation,
model inference, and response processing.

---

## Features

- AI text generation using Qwen2.5-0.5B-Instruct
- Story generation
- Professional email generation
- Text summarization
- Topic explanations
- Creative text generation
- Adjustable generation parameters
- Generation history
- Saved prompts
- Response regeneration
- Download generated responses
- Dark and Light mode
- Live generation statistics
- Custom GENE / AX branding
- Docker support
- Local AI inference without requiring a commercial AI API

---

## Architecture

```text
                    USER
                      |
                      v
              +---------------+
              |  Streamlit UI |
              |    app.py     |
              +-------+-------+
                      |
                      v
              +---------------+
              |   GeneAgent   |
              |   agent.py    |
              +-------+-------+
                      |
                      v
          +-------------------------+
          | Hugging Face Transformers|
          |    Text Generation      |
          +------------+------------+
                       |
                       v
            +----------------------+
            | Qwen2.5-0.5B-Instruct|
            +-----------+----------+
                        |
                        v
                Generated Response
                        |
                        v
                  GENE Streamlit UI

GENE-AI-WRITING-ASSISTANT/
|
├── .streamlit/
│   └── config.toml
|
├── assets/
│   ├── logo.png
│   └── style.css
|
├── app.py
├── agent.py
├── Dockerfile
├── README.md
├── requirements.txt
└── .gitignore

Core Components
app.py

The Streamlit frontend responsible for:

User prompt input
Task selection
Model selection
Generation settings
Chat transcript
History
Saved prompts
Settings
About page
Dark and Light mode
Runtime statistics
Response download
Response regeneration
agent.py

The AI agent controller responsible for:

Prompt validation
Task-specific instruction creation
Hugging Face model loading
Qwen chat-template handling
Text generation
Output extraction
Output cleaning
Structured AgentResponse objects
assets/style.css

Contains the main GENE dashboard styling and interface theme.

| Task          | Purpose                          |
| ------------- | -------------------------------- |
| Story         | Generate short creative stories  |
| Email         | Generate professional emails     |
| Summary       | Summarize provided text          |
| Explanation   | Explain concepts clearly         |
| Creative Text | Generate custom creative content |

Model

GENE uses: Qwen/Qwen2.5-0.5B-Instruct

The model is loaded through: Hugging Face Transformers

The model is downloaded automatically the first time generation is requested and subsequently reused from the local Hugging Face cache.
Generation Controls
GENE provides several generation parameters through the sidebar:

Maximum new tokens
Temperature
Top-p
Top-k
Repetition penalty
Sampling
Number of outputs
Model selection

These parameters allow users to control the behavior of generated
responses.

Local Setup
1. Clone the repository
git clone https://github.com/Akshaya267/GENE-AI-WRITING-ASSISTANT.git
cd GENE-AI-WRITING-ASSISTANT
2. Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Start GENE
streamlit run app.py

The application will be available at: http://localhost:8501

Hardware

GENE can run locally on CPU.

The default Qwen2.5-0.5B model is relatively lightweight, although generation speed depends on the available hardware.

GPU acceleration can be used automatically when a compatible CUDA environment is available.

Docker

Build the Docker image: docker build -t gene-agent .

Run the application: docker run -p 8501:8501 gene-agent

Then open: http://localhost:8501

Customization
Change the Model

The model can be changed from the GENE sidebar or by modifying:

DEFAULT_MODEL_NAME in agent.py.

Larger models require additional system resources.

Add a New Task

Add a new entry to: TASK_INSTRUCTIONS in agent.py.

The task will automatically become available in the Streamlit task
selector.

Tune Generation

Generation defaults are defined in:

GenerationParams

Runtime values can also be adjusted from the sidebar.

Local AI

GENE is designed around local model inference.

The default text-generation workflow does not require a commercial
AI API.

The Qwen model is downloaded from Hugging Face and executed locally through the Transformers library.

Interface

The GENE dashboard includes:

Futuristic dark interface
AX/GENE branding
Generation statistics
Chat-style conversation view
Sidebar controls
Task selector
Model selector
History
Saved prompts
Settings
About section
Dark and Light mode
Project Status

Status: Working MVP / Local AI Writing Assistant

GENE currently provides a complete end-to-end workflow:

User Prompt
     |
     v
Task Selection
     |
     v
Prompt Validation
     |
     v
Task Instruction
     |
     v
Qwen2.5-Instruct
     |
     v
Response Processing
     |
     v
GENE UI

License

MIT License.

Author

Akshaya

GitHub:

https://github.com/Akshaya267
