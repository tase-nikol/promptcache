from __future__ import annotations

import json
import random
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_support(n_intents: int = 200, paraphrases_per_intent: int = 5, seed: int = 7) -> list[dict]:
    random.seed(seed)
    intents = [
        ("password_reset", [
            "How do I reset my password?",
            "I forgot my password. What can I do?",
            "Password reset steps?",
            "Help me change my account password",
            "How to reset my login password?",
        ]),
        ("refund_policy", [
            "What is your refund policy?",
            "Can I get a refund?",
            "How do refunds work?",
            "Do you offer refunds for subscriptions?",
            "Refund rules and timeline?",
        ]),
        ("invoice_download", [
            "How can I download my invoice?",
            "Where do I find invoices?",
            "I need my invoice PDF",
            "Can I get an invoice for last month?",
            "How do I access billing invoices?",
        ]),
        ("change_plan", [
            "How do I change my subscription plan?",
            "Upgrade my plan",
            "Downgrade subscription",
            "Change pricing tier",
            "Switch to annual billing",
        ]),
        ("account_delete", [
            "How do I delete my account?",
            "Close my account permanently",
            "Remove my profile",
            "Delete account steps",
            "Can you erase my account?",
        ]),
    ]

    # Expand to n_intents by templating
    rows: list[dict] = []
    for i in range(n_intents):
        base_intent, examples = random.choice(intents)
        intent_id = f"{base_intent}_{i:04d}"
        # Create paraphrases by reusing + tiny variations
        for j in range(paraphrases_per_intent):
            prompt = examples[j % len(examples)]
            # tiny noise to simulate real users
            if random.random() < 0.35:
                prompt = prompt.replace("?", "") + " please?"
            if random.random() < 0.20:
                prompt = "Hey, " + prompt
            rows.append({
                "workload": "support",
                "intent_id": intent_id,
                "prompt": prompt,
            })
    random.shuffle(rows)
    return rows


def generate_rag(n_intents: int = 200, paraphrases_per_intent: int = 4, seed: int = 11) -> list[dict]:
    random.seed(seed)
    topics = [
        "Redis persistence", "PostgreSQL indexes", "HTTP caching headers",
        "OAuth PKCE", "JWT security", "Docker networking",
        "Vector search HNSW", "Rate limiting", "Idempotency keys",
        "Observability tracing",
    ]
    patterns = [
        "Explain {topic} in simple terms.",
        "How does {topic} work?",
        "Give me a brief explanation of {topic}.",
        "What are common pitfalls of {topic}?",
        "When should I use {topic}?",
    ]

    rows: list[dict] = []
    for i in range(n_intents):
        topic = random.choice(topics)
        intent_id = f"rag_{i:04d}_{topic.replace(' ', '_').lower()}"
        for j in range(paraphrases_per_intent):
            prompt = patterns[(i + j) % len(patterns)].format(topic=topic)
            if random.random() < 0.25:
                prompt += " Include an example."
            if random.random() < 0.15:
                prompt = prompt.replace("brief", "short")
            rows.append({
                "workload": "rag",
                "intent_id": intent_id,
                "prompt": prompt,
            })
    random.shuffle(rows)
    return rows


def generate_creative(n: int = 600, seed: int = 13) -> list[dict]:
    random.seed(seed)
    prompts = [
        "Write a short poem about the sea and time.",
        "Invent a fantasy city name and its history in 5 sentences.",
        "Give me 10 creative startup names for a gardening app.",
        "Write a dialogue between a cat and a robot.",
        "Describe a surreal painting in vivid detail.",
        "Create a 7-day workout plan for beginners.",
        "Generate 5 plot twists for a mystery novel.",
        "Explain quantum mechanics like I'm 5.",
        "Write a motivational speech for a team.",
        "Design a board game idea in 8 bullet points.",
    ]
    rows: list[dict] = []
    for i in range(n):
        # each is its own "intent" (low repetition)
        intent_id = f"creative_{i:04d}"
        p = random.choice(prompts)
        # add variation to increase uniqueness
        if random.random() < 0.50:
            p += f" (style: {random.choice(['funny', 'serious', 'minimalist', 'dramatic'])})"
        if random.random() < 0.25:
            p = "Please " + p[0].lower() + p[1:]
        rows.append({
            "workload": "creative",
            "intent_id": intent_id,
            "prompt": p,
        })
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    _write_jsonl(out_dir / "support.jsonl", generate_support())
    _write_jsonl(out_dir / "rag.jsonl", generate_rag())
    _write_jsonl(out_dir / "creative.jsonl", generate_creative())
    print("Wrote workloads:", out_dir)
