"""Alternating dialogue and run lifecycle, independent of Streamlit."""

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from threading import Condition, Thread
from uuid import uuid4

from .ollama import OllamaError, Reply


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Agent:
    name: str
    model: str
    system_prompt: str


@dataclass(frozen=True)
class Experiment:
    preset: str
    agent_a: Agent
    agent_b: Agent
    initial_message: str
    message_count: int
    temperature: float
    max_tokens: int

    def validate(self) -> None:
        for agent in (self.agent_a, self.agent_b):
            if not all(field.strip() for field in (agent.name, agent.model, agent.system_prompt)):
                raise ValueError("Both agents need a name, a local model, and a system prompt.")
        if not self.initial_message.strip():
            raise ValueError("Enter an initial message for Agent A.")
        if type(self.message_count) is not int or self.message_count < 1:
            raise ValueError("Total messages must be a positive integer.")
        if not isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("Max output tokens must be a positive integer.")


@dataclass(frozen=True)
class Turn:
    number: int
    agent: str
    speaker: str
    timestamp: str
    reply: Reply


@dataclass(frozen=True)
class Run:
    id: str
    config: Experiment
    started_at: str
    updated_at: str
    status: str = "running"
    turns: tuple[Turn, ...] = ()
    error: str | None = None


def agent_messages(run: Run, agent_id: str) -> list[dict[str, str]]:
    config = run.config
    agent = config.agent_a if agent_id == "A" else config.agent_b
    history = [{"role": "system", "content": agent.system_prompt}]
    if agent_id == "B" and run.turns:
        first, *remaining = run.turns
        history.append({
            "role": "user",
            "content": f"Experiment opening message:\n{config.initial_message}\n\nAgent A's first reply:\n{first.reply.text}",
        })
    else:
        history.append({"role": "user", "content": config.initial_message})
        remaining = run.turns
    history.extend(
        {"role": "assistant" if turn.agent == agent_id else "user", "content": turn.reply.text}
        for turn in remaining
    )
    return history


class ConversationController:
    def __init__(self, client, store):
        self._client = client
        self._store = store
        self._changed = Condition()
        self._run: Run | None = None
        self._thread: Thread | None = None
        self._in_flight: str | None = None

    def snapshot(self) -> tuple[Run | None, bool, str | None]:
        with self._changed:
            return self._run, self._thread is not None, self._in_flight

    def start(self, config: Experiment) -> None:
        config.validate()
        with self._changed:
            if self._thread is not None:
                raise ValueError("Wait for the current experiment to finish or stop it first.")
            now = utc_now()
            run = Run(uuid4().hex, config, now, now)
            self._store.save(run)
            self._run = run
            self._thread = Thread(target=self._work, name="agentic-improv", daemon=True)
            try:
                self._thread.start()
            except RuntimeError:
                self._thread = None
                self._fail("The conversation worker could not start. Try again.")

    def pause(self) -> None:
        self._transition({"running"}, "paused")

    def resume(self) -> None:
        self._transition({"paused"}, "running")

    def stop(self) -> None:
        self._transition({"running", "paused"}, "stopped")

    def _transition(self, allowed: set[str], status: str) -> None:
        with self._changed:
            if self._run and self._run.status in allowed:
                self._publish(status=status)
                self._changed.notify_all()

    def _publish(self, **changes) -> None:
        run = replace(self._run, updated_at=utc_now(), **changes)
        self._store.save(run)
        self._run = run

    def _fail(self, message: str) -> None:
        try:
            self._publish(status="failed", error=message)
        except OSError:
            self._run = replace(
                self._run, updated_at=utc_now(), status="failed",
                error=f"{message} The last saved checkpoint is still on disk. Check write access to the runs folder.",
            )

    def _work(self) -> None:
        try:
            while True:
                with self._changed:
                    self._changed.wait_for(lambda: self._run.status != "paused")
                    run = self._run
                    if run.status != "running":
                        return
                    number = len(run.turns) + 1
                    agent_id = "A" if number % 2 else "B"
                    agent = run.config.agent_a if agent_id == "A" else run.config.agent_b
                    self._in_flight = agent_id

                reply = self._client.chat(
                    agent.model, agent_messages(run, agent_id), run.config.temperature, run.config.max_tokens,
                )

                with self._changed:
                    self._in_flight = None
                    if self._run.status == "stopped":
                        return
                    turn = Turn(number, agent_id, agent.name, utc_now(), reply)
                    status = "completed" if number == run.config.message_count else self._run.status
                    self._publish(status=status, turns=(*run.turns, turn))
        except OllamaError as exc:
            self._record_failure(str(exc))
        except OSError:
            self._record_failure("The experiment could not save its latest update.")
        except Exception:
            logging.exception("Conversation worker failed")
            self._record_failure("The conversation worker stopped unexpectedly. See the application terminal.")
        finally:
            with self._changed:
                self._in_flight = None
                self._thread = None

    def _record_failure(self, message: str) -> None:
        with self._changed:
            if self._run.status != "stopped":
                self._fail(message)
