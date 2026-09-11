"""Answer generation: extractive (default), optional OpenAI or Ollama."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

import httpx

from app.core.config import Settings, get_settings
from app.rag.embeddings import QUERY_STOP, tokenize
from app.rag.store import Hit

log = logging.getLogger("rag.generate")

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, question: str, hits: list[Hit]) -> str:
        raise NotImplementedError


def _passages(text: str) -> list[str]:
    """Prefer markdown lines/bullets/table rows over whole-section blobs."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells or cells[0].lower() in {"severity", ""}:
                continue
            if set("".join(cells)) <= set("-: "):
                continue
            line = " — ".join(c for c in cells if c)
        elif set(line) <= set("-|: "):
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if line.startswith("_") or len(line) < 12:
            continue
        if ". " in line:
            out.extend(p.strip() for p in _SENTENCE.split(line) if p.strip())
        else:
            out.append(line)
    return out


class ExtractiveGenerator(Generator):
    """Stitch the most question-relevant sentences from retrieved chunks.

    No LLM required. Grounded in ingested text only — preferred Day 11 default.
    """

    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return (
                "I do not have that in the ingested company docs yet. "
                "POST /ingest with files under data/sample_docs/ and try again."
            )
        q_terms = set(tokenize(question)) - QUERY_STOP
        wants_number = bool(
            re.search(
                r"how many|how much|stipend|response time|\bsla\b|\bptos?\b",
                question.lower(),
            )
        )
        selected: list[tuple[str, str]] = []
        seen: list[str] = []
        for hit in hits:
            passages = _passages(hit.text)
            if not passages:
                compact = re.sub(r"\s+", " ", hit.text).strip()
                passages = [compact[:320]] if compact else []
            chunk_overlap = len(q_terms & set(tokenize(f"{hit.title} {hit.text}")))
            ranked: list[tuple[float, str]] = []
            for passage in passages:
                key = passage.lower()
                if key in seen:
                    continue
                overlap = len(q_terms & set(tokenize(passage)))
                score = overlap + 0.5 * chunk_overlap + 0.3 * hit.score
                if wants_number and re.search(r"[\d$]", passage):
                    score += 1.2
                if len(passage) < 28:
                    score -= 0.2
                ranked.append((score, passage))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if not ranked:
                continue
            take_n = 2
            for _, passage in ranked[:take_n]:
                if passage.lower() in seen:
                    continue
                selected.append((passage, hit.title))
                seen.append(passage.lower())
            if wants_number and not any(re.search(r"[\d$]", text) for text, _ in selected):
                numeric = next(
                    (p for _, p in ranked if re.search(r"[\d$]", p) and p.lower() not in seen),
                    None,
                )
                if numeric:
                    selected.append((numeric, hit.title))
                    seen.append(numeric.lower())
            if selected:
                break
        if not selected:
            selected = [(re.sub(r"\s+", " ", hits[0].text).strip()[:320], hits[0].title)]
        body = " ".join(text for text, _ in selected)
        titles: list[str] = []
        for _, title in selected:
            if title not in titles:
                titles.append(title)
        return f"{body}\n\nSources: {'; '.join(titles)}."


class OpenAIGenerator(Generator):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return ExtractiveGenerator().generate(question, hits)
        context = "\n\n".join(
            f"[{hit.title} | {hit.source}]\n{hit.text}" for hit in hits
        )
        system = (
            "You are the Vision AI Ops company knowledge assistant. "
            "Answer only from the provided context. If the context is missing "
            "the answer, say you do not know. Cite source titles. "
            "This is a fictional portfolio demo — never invent secrets."
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()


class OllamaGenerator(Generator):
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return ExtractiveGenerator().generate(question, hits)
        context = "\n\n".join(
            f"[{hit.title} | {hit.source}]\n{hit.text}" for hit in hits
        )
        prompt = (
            "Answer only from the context. If missing, say you do not know. "
            "Cite source titles.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._url}/api/generate", json=payload)
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()


def get_generator(settings: Settings | None = None) -> Generator:
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider == "extractive":
        return ExtractiveGenerator()
    if provider == "openai":
        return OpenAIGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout=settings.ai_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaGenerator(
            url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=settings.ai_timeout_seconds,
        )
    raise ValueError(f"unknown llm provider: {provider}")
