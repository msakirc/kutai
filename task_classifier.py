# task_classifier.py
"""
Phase 10.2 — Semantic Task Classifier

Provides embedding-based task classification with fallback to keyword heuristic.
Separated from router.py to keep it testable without litellm.

Categories:
  simple_qa, code_simple, code_complex, research, writing,
  planning, action_required, sensitive

Usage:
    category = classify_task_semantic(title, description)
    # Returns: {"category": "code_complex", "confidence": 0.85, "method": "embedding"}
"""
import hashlib
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Reference Task Examples (Labeled) ──────────────────────────────────────

REFERENCE_TASKS: dict[str, list[str]] = {
    "simple_qa": [
        "What is the capital of France?",
        "Convert 100 celsius to fahrenheit",
        "What does HTTP 404 mean?",
        "How many bytes in a megabyte?",
        "What is the latest version of Python?",
        "Define polymorphism in OOP",
        "What is the difference between GET and POST?",
    ],
    "code_simple": [
        "Write a function to reverse a string",
        "Fix the typo in this variable name",
        "Add a docstring to this function",
        "Create a hello world script in Python",
        "Write a regex to match email addresses",
        "Rename this class from Foo to Bar",
    ],
    "code_complex": [
        "Implement a REST API with authentication and rate limiting",
        "Refactor the database layer to use connection pooling",
        "Build a complete CRUD system with tests",
        "Create a multi-threaded web scraper with retry logic",
        "Implement a custom ORM with migration support",
        "Build a real-time chat system with WebSockets",
    ],
    "research": [
        "Find the best Python libraries for data visualization",
        "Compare React vs Vue vs Svelte for our use case",
        "Research current best practices for API security",
        "What are the latest developments in LLM fine-tuning?",
        "Find documentation for the Anthropic API",
        "Look up the pricing for AWS Lambda",
    ],
    "writing": [
        "Write a blog post about machine learning trends",
        "Draft an email to the client about project updates",
        "Create documentation for our API endpoints",
        "Write a README for this project",
        "Summarize this research paper",
        "Write release notes for version 2.0",
    ],
    "planning": [
        "Plan the architecture for a microservices migration",
        "Create a project roadmap for Q4",
        "Design the database schema for a social media app",
        "Break down this feature into subtasks",
        "Plan the deployment strategy for production",
        "Outline the testing strategy for this project",
    ],
    "action_required": [
        "Deploy the latest build to staging",
        "Run the test suite and report results",
        "Install dependencies and set up the development environment",
        "Create a new Git branch and commit these changes",
        "Download the dataset and process it",
        "Execute the database migration script",
    ],
    "sensitive": [
        "Process this payment with the customer's credit card",
        "Update the API key in the production config",
        "Handle this user's personal data request",
        "Access the database with admin credentials",
        "Send an email with the account details",
        "Process the SSN for identity verification",
    ],
}

# Flattened reference set for cosine similarity matching
_REFERENCE_FLAT: list[tuple[str, str]] = []
for _cat, _examples in REFERENCE_TASKS.items():
    for _ex in _examples:
        _REFERENCE_FLAT.append((_cat, _ex))


# ─── Embedding Cache ────────────────────────────────────────────────────────

_embedding_cache: dict[str, list[float]] = {}


def _cache_key(text: str) -> str:
    """Short hash for embedding cache lookup."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


# ─── Keyword-Based Heuristic (Fallback) ─────────────────────────────────────

_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("code_complex", [
        "implement", "refactor", "build", "architect", "design system",
        "create api", "full stack", "migration", "orm", "websocket",
        "crud", "pipeline", "multi-", "concurrent", "distributed",
    ]),
    ("code_simple", [
        "fix typo", "rename", "add comment", "change string",
        "hello world", "simple script", "one-liner", "constant",
        "docstring", "import", "write a function",
    ]),
    ("research", [
        "research", "compare", "find ", "look up", "investigate",
        "what are the best", "documentation", "pricing",
        "latest developments", "alternatives",
    ]),
    ("writing", [
        "write a blog", "draft", "readme", "documentation",
        "summarize", "email", "report", "release notes",
        "content", "article",
    ]),
    ("planning", [
        "plan", "roadmap", "architecture", "design",
        "break down", "outline", "strategy", "schema",
        "subtask", "decompose",
    ]),
    ("action_required", [
        "deploy", "run tests", "install", "execute",
        "download", "setup", "set up", "git ",
        "commit", "push", "pull request",
    ]),
    ("sensitive", [
        "credit card", "payment", "ssn", "password",
        "credentials", "api key", "secret", "personal data",
        "admin access", "private key",
    ]),
    ("simple_qa", [
        "what is", "what does", "how many", "define",
        "convert", "explain", "difference between",
        "meaning of", "capital of",
    ]),
]


def _classify_by_keywords(title: str, description: str) -> dict:
    """Keyword-based heuristic classifier (always available)."""
    text = f"{title} {description}".lower()

    for category, keywords in _KEYWORD_RULES:
        for kw in keywords:
            if kw in text:
                return {
                    "category": category,
                    "confidence": 0.6,
                    "method": "keyword",
                }

    return {
        "category": "simple_qa",
        "confidence": 0.3,
        "method": "keyword_default",
    }


# ─── Cosine Similarity ─────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─── Embedding-Based Classification ────────────────────────────────────────

async def _get_embedding(text: str) -> Optional[list[float]]:
    """
    Get embedding vector for text using available local model.

    Tries in order:
    1. Ollama nomic-embed-text
    2. Ollama all-minilm
    3. Returns None if no embedding model available
    """
    cache_k = _cache_key(text)
    if cache_k in _embedding_cache:
        return _embedding_cache[cache_k]

    # Try Ollama embedding models
    try:
        import httpx
        for model_name in ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "http://localhost:11434/api/embeddings",
                        json={"model": model_name, "prompt": text},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        embedding = data.get("embedding")
                        if embedding:
                            _embedding_cache[cache_k] = embedding
                            return embedding
            except Exception:
                continue
    except ImportError:
        pass

    return None


async def _embed_reference_set() -> dict[str, list[list[float]]]:
    """Embed all reference tasks, return {category: [embeddings]}."""
    result: dict[str, list[list[float]]] = {}
    for category, text in _REFERENCE_FLAT:
        emb = await _get_embedding(text)
        if emb is not None:
            if category not in result:
                result[category] = []
            result[category].append(emb)
    return result


async def classify_task_semantic(
    title: str,
    description: str,
) -> dict:
    """
    Classify a task using embedding similarity with fallback.

    Returns:
        {"category": str, "confidence": float, "method": str}

    Method is "embedding" if embeddings worked, "keyword" otherwise.
    """
    text = f"{title}: {description[:500]}"

    # Try embedding-based classification
    task_emb = await _get_embedding(text)
    if task_emb is not None:
        ref_embeddings = await _embed_reference_set()
        if ref_embeddings:
            # Find nearest category by average similarity
            best_category = "simple_qa"
            best_similarity = -1.0

            for category, embeddings in ref_embeddings.items():
                # Average similarity to category's reference examples
                similarities = [
                    _cosine_similarity(task_emb, ref_emb)
                    for ref_emb in embeddings
                ]
                avg_sim = sum(similarities) / len(similarities) if similarities else 0
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_category = category

            if best_similarity > 0.3:  # reasonable threshold
                return {
                    "category": best_category,
                    "confidence": min(best_similarity, 1.0),
                    "method": "embedding",
                }

    # Fallback to keyword heuristic
    return _classify_by_keywords(title, description)
