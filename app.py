"""Streamlit entry point for AgenticImprov."""

from pathlib import Path

import streamlit as st

from agentic_improv.controller import Agent, ConversationController, Experiment
from agentic_improv.ollama import OLLAMA_URL, OllamaClient, OllamaError
from agentic_improv.presets import PRESETS
from agentic_improv.storage import RunStore, csv_export, json_export, run_document


RUNS_DIRECTORY = Path(__file__).resolve().parent / "runs"


@st.cache_resource
def local_lab() -> tuple[OllamaClient, RunStore, ConversationController]:
    client = OllamaClient()
    store = RunStore(RUNS_DIRECTORY)
    return client, store, ConversationController(client, store)


def load_preset() -> None:
    st.session_state.update(PRESETS[st.session_state.preset])


def model_picker(label: str, key: str, models: list[str], preferred: str) -> str | None:
    if st.session_state.get(key) not in models:
        st.session_state[key] = preferred if preferred in models else next(iter(models), None)
    return st.selectbox(label, models, key=key, disabled=not models, placeholder="No local models")


def setup(client: OllamaClient) -> tuple[Experiment, bool]:
    with st.sidebar:
        st.header("Ollama")
        st.caption(OLLAMA_URL)
        refresh = st.button("Refresh models")
        if "models" not in st.session_state or refresh:
            try:
                st.session_state.models = client.list_models()
                st.session_state.model_error = None
            except OllamaError as exc:
                st.session_state.models = []
                st.session_state.model_error = str(exc)
        models = st.session_state.models
        if st.session_state.model_error:
            st.error(st.session_state.model_error)
        elif not models:
            st.warning("No local models found. Run `ollama pull qwen3:4b`, then refresh.")
        else:
            st.success(f"Local models available · {len(models)}")

        st.header("Experiment settings")
        preset = st.selectbox("Preset", list(PRESETS), key="preset", on_change=load_preset)
        count = st.number_input("Total messages", min_value=1, value=8, step=1, help="Counts agent replies. Agent A replies first.")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens = st.number_input("Max output tokens", min_value=1, value=256, step=1)

    if "name_a" not in st.session_state:
        load_preset()
    st.header("Agent setup")
    with st.expander("Models, names, and prompts", expanded=True):
        left, right = st.columns(2)
        with left:
            st.subheader("Agent A")
            model_a = model_picker("Agent A model", "model_a", models, "qwen3:4b")
            name_a = st.text_input("Agent A name", key="name_a")
            prompt_a = st.text_area("Agent A system prompt", key="prompt_a", height=175)
        with right:
            st.subheader("Agent B")
            model_b = model_picker("Agent B model", "model_b", models, "llama3.2:3b")
            name_b = st.text_input("Agent B name", key="name_b")
            prompt_b = st.text_area("Agent B system prompt", key="prompt_b", height=175)
        initial = st.text_area("Initial message to Agent A", key="initial_message", height=95)
    config = Experiment(
        preset, Agent(name_a.strip(), model_a or "", prompt_a), Agent(name_b.strip(), model_b or "", prompt_b),
        initial, count, temperature, max_tokens,
    )
    return config, bool(model_a and model_b)


def token_label(value: int | None) -> str:
    return "—" if value is None else str(value)


def transcript(data: dict, key_prefix: str) -> None:
    with st.container(height=400):
        with st.chat_message("user", avatar="📝"):
            st.text("Initial message")
            st.text(data["config"]["initial_message"])
        for turn in data["turns"]:
            reply = turn["reply"]
            avatar = "🅰️" if turn["agent"] == "A" else "🅱️"
            with st.chat_message(f"Agent {turn['agent']}", avatar=avatar):
                st.text(f"{turn['number']}. {turn['speaker']} · Agent {turn['agent']} · {reply['model']}")
                st.text(reply["text"])
                st.caption(
                    f"{reply['latency_ms'] / 1000:.2f} s · "
                    f"{token_label(reply['input_tokens'])} input tokens · "
                    f"{token_label(reply['output_tokens'])} output tokens"
                )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download JSON", json_export(data), f"agentic-improv-{data['id']}.json", "application/json",
        key=f"{key_prefix}_json", on_click="ignore",
    )
    downloads[1].download_button(
        "Download CSV", csv_export(data).encode("utf-8-sig"), f"agentic-improv-{data['id']}.csv", "text/csv",
        key=f"{key_prefix}_csv", on_click="ignore",
    )


def control(action) -> None:
    st.session_state.control_error = None
    try:
        action()
    except ValueError as exc:
        st.session_state.control_error = str(exc)
    except OSError:
        st.session_state.control_error = "Could not write the experiment to the runs folder. Check folder permissions and free disk space."


def live_conversation(controller: ConversationController, config: Experiment, models_ready: bool, polling: bool) -> None:
    @st.fragment(run_every=1.0 if polling else None)
    def panel():
        run, busy, in_flight = controller.snapshot()
        if polling and not busy:
            st.rerun()
        status = run.status if run else "ready"
        buttons = st.columns(4)
        if buttons[0].button("Start", type="primary", disabled=busy or not models_ready, width="stretch"):
            control(lambda: controller.start(config))
            if not st.session_state.control_error:
                st.rerun()
        buttons[1].button("Pause", disabled=status != "running", on_click=control, args=(controller.pause,), width="stretch")
        buttons[2].button("Resume", disabled=status != "paused", on_click=control, args=(controller.resume,), width="stretch")
        buttons[3].button("Stop", disabled=status not in {"running", "paused"}, on_click=control, args=(controller.stop,), width="stretch")
        if error := st.session_state.get("control_error"):
            st.error(error)
        if run is None:
            st.info("Choose local models for both agents, then start an experiment.")
            return
        st.progress(len(run.turns) / run.config.message_count, text=f"{status.title()} · {len(run.turns)} of {run.config.message_count} replies")
        if run.error:
            st.error(run.error)
        if in_flight and status == "running":
            st.caption(f"Waiting for Agent {in_flight} to reply…")
        elif in_flight and status == "paused":
            st.caption(f"Agent {in_flight} is finishing its current reply.")
        elif busy and status == "stopped":
            st.caption("Waiting for the outstanding Ollama request to finish.")
        transcript(run_document(run), "live")

    panel()


def saved_experiments(store: RunStore, current_id: str | None) -> None:
    with st.expander("Saved experiments"):
        runs = {data["id"]: data for data in store.archived_runs(current_id)}
        if not runs:
            st.caption("Earlier experiments will appear here.")
            return

        def label(run_id: str) -> str:
            data = runs[run_id]
            started = data["started_at"][:19].replace("T", " ")
            return f"{started} UTC · {data['config']['preset']} · {run_id[:8]}"

        selected = st.selectbox("Experiment", list(runs), index=None, format_func=label, placeholder="Choose a saved experiment")
        if selected:
            data = runs[selected]
            st.caption(f"{data['status'].title()} · {len(data['turns'])} of {data['config']['message_count']} replies")
            if data.get("error"):
                st.warning(data["error"])
            transcript(data, "archive")


def main() -> None:
    st.set_page_config(page_title="AgenticImprov", layout="centered")
    st.title("AgenticImprov")
    st.caption("Two autonomous agents running on local Ollama models.")
    client, store, controller = local_lab()
    config, models_ready = setup(client)
    run, busy, _ = controller.snapshot()
    st.header("Conversation")
    live_conversation(controller, config, models_ready, busy)
    saved_experiments(store, run.id if run else None)


if __name__ == "__main__":
    main()
