import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from queue import Empty, Queue
from time import monotonic, sleep
from unittest.mock import patch

from agentic_improv.controller import Agent, ConversationController, Experiment
from agentic_improv.ollama import OllamaClient, OllamaError, Reply
from agentic_improv.storage import RunStore, csv_export, run_document


CONFIG = Experiment("Test", Agent("Lamp", "qwen3:4b", "Lamp instructions"),
                    Agent("Curtain", "llama3.2:3b", "Curtain instructions"), "Opening", 4, 0.4, 48)


class ControlledClient:
    def __init__(self):
        self.calls = Queue()
        self.results = Queue()

    def chat(self, model, messages, temperature, max_tokens):
        self.calls.put((model, messages, temperature, max_tokens))
        result = self.results.get(timeout=3)
        if isinstance(result, Exception):
            raise result
        return Reply(result, model, 25.0, 12, 4)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = RunStore(Path(directory.name))
        self.client = ControlledClient()
        self.controller = ConversationController(self.client, self.store)
        self.addCleanup(self.finish_worker)

    def finish_worker(self):
        self.controller.stop()
        self.client.results.put("Cleanup")
        self.wait_until(lambda: not self.controller.snapshot()[1])

    def wait_until(self, predicate):
        deadline = monotonic() + 3
        while not predicate():
            if monotonic() >= deadline:
                self.fail("Controller did not reach the expected state.")
            sleep(0.005)

    def saved(self):
        run, _, _ = self.controller.snapshot()
        return json.loads((self.store.directory / f"{run.id}.json").read_text(encoding="utf-8"))

    def test_alternation_context_settings_and_exports(self):
        self.controller.start(CONFIG)
        calls = []
        for text in ("A1", "B1", "A2", "B2"):
            calls.append(self.client.calls.get(timeout=3))
            self.client.results.put(text)
        self.wait_until(lambda: not self.controller.snapshot()[1])
        run, _, _ = self.controller.snapshot()
        self.assertEqual(run.status, "completed")
        self.assertEqual([call[0] for call in calls], ["qwen3:4b", "llama3.2:3b"] * 2)
        self.assertTrue(all(call[2:] == (0.4, 48) for call in calls))
        self.assertEqual(calls[2][1], [
            {"role": "system", "content": "Lamp instructions"}, {"role": "user", "content": "Opening"},
            {"role": "assistant", "content": "A1"}, {"role": "user", "content": "B1"},
        ])
        b_history = calls[3][1]
        self.assertEqual(b_history[0], {"role": "system", "content": "Curtain instructions"})
        self.assertEqual(b_history[1]["role"], "user")
        self.assertIn("Opening", b_history[1]["content"])
        self.assertIn("A1", b_history[1]["content"])
        self.assertEqual(b_history[2:], [{"role": "assistant", "content": "B1"}, {"role": "user", "content": "A2"}])
        data = self.saved()
        self.assertEqual(data, json.loads(json.dumps(run_document(run))))
        rows = list(csv.DictReader(io.StringIO(csv_export(data))))
        self.assertEqual([row["message"] for row in rows], ["Opening", "A1", "B1", "A2", "B2"])
        self.assertEqual(rows[-1]["output_tokens"], "4")

    def test_pause_checkpoints_in_flight_reply_and_resume_continues(self):
        self.controller.start(CONFIG)
        self.client.calls.get(timeout=3)
        self.controller.pause()
        self.client.results.put("A1")
        self.wait_until(lambda: len(self.controller.snapshot()[0].turns) == 1)
        self.assertEqual(self.saved()["status"], "paused")
        with self.assertRaises(Empty):
            self.client.calls.get(timeout=0.05)
        self.controller.resume()
        self.assertEqual(self.client.calls.get(timeout=3)[0], "llama3.2:3b")

    def test_stop_keeps_checkpoint_and_discards_outstanding_reply(self):
        self.controller.start(CONFIG)
        self.client.calls.get(timeout=3)
        self.controller.stop()
        self.assertEqual(self.saved()["status"], "stopped")
        with self.assertRaises(ValueError):
            self.controller.start(CONFIG)
        self.client.results.put("Discard this reply")
        self.wait_until(lambda: not self.controller.snapshot()[1])
        self.assertEqual(self.saved()["turns"], [])
        self.assertTrue(self.client.calls.empty())

    def test_failure_preserves_completed_replies(self):
        self.controller.start(CONFIG)
        self.client.calls.get(timeout=3)
        self.client.results.put("A1")
        self.client.calls.get(timeout=3)
        self.client.results.put(OllamaError("Ollama returned no visible reply."))
        self.wait_until(lambda: not self.controller.snapshot()[1])
        saved = self.saved()
        self.assertEqual(saved["status"], "failed")
        self.assertIn("no visible reply", saved["error"])
        self.assertEqual([turn["reply"]["text"] for turn in saved["turns"]], ["A1"])

    def test_reopened_interrupted_run_exposes_last_checkpoint(self):
        self.controller.start(CONFIG)
        self.client.calls.get(timeout=3)
        self.client.results.put("A1")
        self.client.calls.get(timeout=3)
        archive = RunStore(self.store.directory).archived_runs()
        self.assertEqual(archive[0]["status"], "interrupted")
        self.assertEqual(archive[0]["turns"][0]["reply"]["text"], "A1")


class OllamaTests(unittest.TestCase):
    def test_model_discovery_filters_remote_entries(self):
        client = OllamaClient()
        entries = [{"name": "qwen3:4b"}, {"model": "llama3.2:3b"}, {"name": "example:cloud"},
                   {"name": "alias:latest", "remote_host": "example.invalid"}]
        with patch.object(client, "_request", return_value={"models": entries}):
            self.assertEqual(client.list_models(), ["llama3.2:3b", "qwen3:4b"])

    def test_thinking_is_disabled_and_filtered(self):
        client = OllamaClient()
        response = {"done": True, "message": {"content": json.dumps({"reply": "<think>hidden</think>Visible"}), "thinking": "hidden too"},
                    "prompt_eval_count": 21, "eval_count": 7}
        messages = [{"role": "system", "content": "Keep it short."}, {"role": "user", "content": "Hello"}]
        with patch.object(client, "_installed_models", return_value={"qwen3:4b": {}}), patch.object(client, "_request", return_value=response) as request:
            reply = client.chat("qwen3:4b", messages, 0.4, 48)
        self.assertEqual(reply.text, "Visible")
        self.assertEqual((reply.input_tokens, reply.output_tokens), (21, 7))
        self.assertIs(request.call_args.args[1]["think"], False)
        self.assertEqual(request.call_args.args[1]["options"], {"temperature": 0.4, "num_predict": 48})
        self.assertEqual(request.call_args.args[1]["format"]["required"], ["reply"])
        self.assertEqual(messages[0]["content"], "Keep it short.")

    def test_empty_response_exposes_actionable_error_without_thinking(self):
        client = OllamaClient()
        response = {"done": True, "done_reason": "length", "message": {"content": "Unmarked private reasoning", "thinking": "private"}}
        messages = [{"role": "system", "content": "Keep it short."}]
        with patch.object(client, "_installed_models", return_value={"qwen3:4b": {}}), patch.object(client, "_request", return_value=response):
            with self.assertRaisesRegex(OllamaError, "Increase Max output tokens") as error:
                client.chat("qwen3:4b", messages, 0.4, 48)
        self.assertNotIn("private", str(error.exception))


if __name__ == "__main__":
    unittest.main()
