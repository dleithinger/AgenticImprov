# AgenticImprov

AgenticImprov is a local Streamlit dashboard for running alternating conversations between two Ollama agents. Each agent has its own model, name, and system prompt. Inference uses the Ollama REST API at `http://localhost:11434`.

## Windows setup

1. Install [Python 3.11 or newer](https://www.python.org/downloads/windows/) and [Ollama for Windows](https://ollama.com/download/windows).
2. Open Ollama so its local server is running. In PowerShell, install any models you still need and check the list:

   ```powershell
   ollama pull qwen3:4b
   ollama pull llama3.2:3b
   ollama list
   ```

3. Double-click `start_windows.bat` in the project root and open [http://localhost:8501](http://localhost:8501). The launcher creates `.venv` and installs Streamlit on first use. Model downloads and dependency installation need internet access during setup. Experiments run locally after setup.

To launch manually from PowerShell in the project root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run an experiment

1. Choose **Collaborative Task (Lamp/Curtain)**, **Protocol Negotiation**, or **Free Conversation**. Switching presets loads their default names, prompts, and initial message into the editable fields.
2. Select a local chat model for each agent and edit the fields as needed. When available, `qwen3:4b` and `llama3.2:3b` are selected by default. Use **Refresh models** after pulling a model or restarting Ollama.
3. Set **Total messages**, **Temperature**, and **Max output tokens**. Total messages counts generated replies, excluding the initial message. Click **Start**. Agent A replies first, followed by Agent B, then the agents alternate. Settings are recorded when the experiment starts. Each agent receives its own system prompt and the visible conversation so far.
4. Replies appear as each turn completes. Latency measures the local chat request and includes model loading when applicable. Token counts come from Ollama. A dash indicates unavailable counts.

**Pause** lets the pending reply finish and saves it before pausing. **Resume** continues with the next speaker. **Stop** records the stopped experiment immediately and discards any pending reply. The next experiment can start once that Ollama request returns or times out. Closing the browser leaves the experiment running while the Streamlit server stays open.

The app requests thinking mode off and filters thinking fields and inline thinking blocks. For Qwen3 it constrains generation to a structured reply field to accommodate model templates that open a thinking block even when thinking is disabled. The output token cap includes the structured reply delimiters.

## Saved experiments

Every experiment gets a `runs/<id>.json` file at startup. The app updates it atomically after each reply and state change. Use **Download JSON** for settings, status, and transcript, or **Download CSV** for the initial message, replies, and per-turn metrics. Earlier experiments appear under **Saved experiments**. If the server closes mid-run, reopening an unfinished experiment shows **Interrupted** and its last saved replies for viewing and export.

## Troubleshooting

- **Connection failed.** Open the Ollama desktop app, or run `ollama serve` in another terminal if it is not running. Confirm `ollama list` works, then click **Refresh models**.
- **Empty reply or request failure.** The app preserves completed replies and marks the run **Failed**. If Ollama reached the output limit, increase **Max output tokens** before starting again. For server errors, check the Ollama terminal or server log.
- **Port 8501 in use.** Close the older Streamlit app or launch with `--server.port 8502` and open that port in your browser.

## Development

`app.py` renders the dashboard. `agentic_improv/controller.py` manages conversation state and turn order, `ollama.py` handles REST requests and reply filtering, `storage.py` handles checkpoints and exports, and `presets.py` defines the scenario defaults.

The controller and REST adapter tests run with Python's standard library and use test doubles for Ollama:

```powershell
python -m compileall -q app.py agentic_improv tests
python -m unittest discover -s tests -v
```

API references: [Ollama model list](https://docs.ollama.com/api/tags), [chat](https://docs.ollama.com/api/chat), and [thinking controls](https://docs.ollama.com/capabilities/thinking).
