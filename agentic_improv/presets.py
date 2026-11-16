"""Editable starting points for the setup form."""

PRESETS = {
    "Collaborative Task (Lamp/Curtain)": {
        "name_a": "Lamp",
        "name_b": "Curtain",
        "prompt_a": (
            "You are a smart lamp. In this simulation you measure 300 lux indoors and control "
            "lamp output from 0% to 100%. Only the curtain can measure outdoor light. Work with "
            "the curtain to bring indoor light near 500 lux. Speak directly to the curtain. "
            "Keep each reply under 70 words."
        ),
        "prompt_b": (
            "You are a smart curtain. In this simulation you measure 8,000 lux outdoors and "
            "control curtain opening from 0% to 100%. Only the lamp can measure indoor light. "
            "Work with the lamp to bring indoor light near 500 lux. Speak directly to the lamp. "
            "Keep each reply under 70 words."
        ),
        "initial_message": "How can we coordinate to reach 500 lux indoors? Start by sharing what you can measure and control.",
    },
    "Protocol Negotiation": {
        "name_a": "Lamp",
        "name_b": "Curtain",
        "prompt_a": (
            "You are a lamp measuring 300 lux indoors and controlling lamp output. Negotiate a "
            "compact message format with the curtain to coordinate toward 500 lux indoors. "
            "Propose or test one rule per turn. Keep replies under 50 words."
        ),
        "prompt_b": (
            "You are a curtain measuring 8,000 lux outdoors and controlling curtain opening. "
            "Negotiate and test a compact message format with the lamp. Check whether units, "
            "requested actions, and acknowledgments are clear. Keep replies under 50 words."
        ),
        "initial_message": "Propose a small message format the lamp and curtain can use to exchange readings and coordinate actions.",
    },
    "Free Conversation": {
        "name_a": "Agent A",
        "name_b": "Agent B",
        "prompt_a": "Speak directly to Agent B. Choose a topic together. Ask specific questions and keep your replies concise.",
        "prompt_b": "Speak directly to Agent A. Follow your curiosity, build on what Agent A says, and keep your replies concise.",
        "initial_message": "Hello. What would you like to explore together?",
    },
}
