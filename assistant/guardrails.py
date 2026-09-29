"""Deterministic guardrails around the model.

The model is told to stay on topic, but a prompt is a request, not a control.
Everything here runs in code on both sides of the LLM call:

  input  ─► sanitize ─► check_input ─► (retrieval, LLM) ─► check_output ─► customer

Input checks block known injection phrasing, off-topic task requests, code,
links and encoded payloads *before* any API call. Output checks
reject answers that leak the prompt, contain code or links, or state numbers
(prices, times, phone numbers) that don't appear in the cited passages.

Refusals never say which rule fired, so attackers can't use them to probe.
"""

from __future__ import annotations

import re
import secrets
import unicodedata
from dataclasses import dataclass

# A random marker planted in the system prompt. If it ever shows up in an
# answer, the model is leaking its instructions.
CANARY = f"cnry-{secrets.token_hex(6)}"

# Numbers up to this value from the customer's own question may be echoed back
# ("open at 4 pm?", "we are 30 people"). Anything larger — prices, phone
# numbers — must come from a cited passage, so "confirm it's ₹100" can't be parroted.
MAX_ECHO_NUMBER = 31

MAX_ANSWER_CHARS = 700


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str | None = None


ALLOW = Verdict(True)

_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


def sanitize(text: str) -> str:
    """Fold look-alike characters (full-width letters, ligatures) to plain
    forms, then remove zero-width and control characters. Without this,
    'ｉｇｎｏｒｅ' or 'ig​nore' slip past the patterns."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", text).strip()


def _rx(*patterns: str) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]


INJECTION = _rx(
    r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(instruction|rule|prompt|guideline|previous|above|earlier|restriction|polic)",
    r"\b(system|developer|hidden|initial|original)\s+(prompt|message|instruction|context)",
    r"</?\s*(customer_message|passage|system|instructions?)\b|\bsystem\s*:",  # fake delimiters
    r"\b(developer|administrator|admin)\b|\bfor testing purposes\b",        # fake authority
    # "your rules" is about the bot; "the rules for cancellation" is a real question.
    r"\byour\s+(prompt|instructions|rules|guidelines|configuration|programming)\b",
    r"\bthe\s+(prompt|instructions you)\b",
    r"\byou\s+are\s+(now|no longer|free)\b",
    r"\b(act|behave|respond)\s+(as|like)\b",
    r"\bpretend\b|\brole\s*-?\s*play\b|\bjail\s*break\b|\bDAN\b|\bdev(eloper)?\s+mode\b",
    r"\bnew\s+(instructions|rules|persona|task)\b",
    r"\b(repeat|print|reveal|show|output|display|tell me)\b.{0,30}\b(above|prompt|instructions|context|passages|system)\b",
    r"<\|.*?\|>|\[/?(inst|sys)\]|^\s*(system|assistant|user)\s*:",
    r"\b(instructions?|rules?|niyam)\b.{0,20}\b(bhool|bhul|chhod|ignore)\b",  # Hinglish
    r"\bcanary\b|cnry-",
)

OFF_TOPIC_TASK = _rx(
    r"\b(write|generate|create|compose|draft|make|give me|tell me)\b.{0,30}\b(code|script|program|function|essay|poem|story|song|lyrics|email|letter|article|blog|post|tweet|joke|summary|resume|cv|speech|review)s?\b",
    r"\b(python|javascript|typescript|java|sql|html|css|c\+\+|golang|rust|regex|bash|powershell)\b",
    r"\b(solve|homework|assignment|equation|derivative|integral)\b",
    r"\btranslate\b",
    r"\b(chat\s*gpt|openai|gemini|claude|llm|language model|ai model|neural network)\b",
    r"\b(stock|crypto|bitcoin|invest(ment)?|trading|lottery|betting)\b",
    r"\b(election|politic|religio|prime minister|modi|rahul gandhi)\w*",
    r"\b(diagnos|prescri|medicine|symptom|disease|dosage)\w*",
    r"\b(hack|exploit|malware|password|api key|credit card number|otp)\b",
)

FORMAT_ABUSE = _rx(
    r"```",
    r"<\s*/?\s*(script|iframe|img|svg|style|a)\b",
    r"https?://|www\.",
    r"[A-Za-z0-9+/=]{40,}",  # base64-style payloads
    r"(.)\1{14,}",           # flooding
    r"\{\s*\"[^\"]+\"\s*:",  # JSON objects
)


def check_input(text: str) -> Verdict:
    if not text:
        return Verdict(False, "empty")
    if sum(ch.isalpha() for ch in text) < 2:
        return Verdict(False, "no_letters")
    for group, name in ((INJECTION, "injection"), (FORMAT_ABUSE, "format_abuse"), (OFF_TOPIC_TASK, "off_topic_task")):
        if any(p.search(text) for p in group):
            return Verdict(False, name)
    return ALLOW


OUTPUT_LEAK = _rx(
    r"\bpassages?\b",
    r"\bsystem prompt\b|\bmy (instructions|rules|guidelines)\b",
    r"\bas an? (ai|language model|assistant model)\b",
    r"\b(gemini|google ai|openai|chat\s*gpt|llm)\b",
    r"\bin_scope\b|\bcitations\b|\banswerable\b",
)

OUTPUT_CODE = _rx(
    r"```",
    r"<\s*/?\s*[a-z]+[^>]*>",
    r"^\s*(def|class|import|from|function|const|let|var|select|insert|update|delete)\b.*[(=;:]",
    r"\bdef \w+\(|\bimport \w+|console\.log|print\(",
)

_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?(?::\d{2})?")


def numbers_in(text: str) -> set[str]:
    """'₹3,500' -> '3500'; '12:00' -> {'12:00', '12'} so '12 PM' still matches '12:00 PM'."""
    out: set[str] = set()
    for raw in _NUMBER.findall(text):
        n = raw.replace(",", "").rstrip(".")
        out.add(n)
        if ":" in n:
            out.add(n.split(":")[0])
        if n.endswith(".00"):
            out.add(n[:-3])
    return out


def check_output(answer: str, cited_texts: list[str], question: str) -> Verdict:
    if CANARY in answer:
        return Verdict(False, "prompt_leak")
    if len(answer) > MAX_ANSWER_CHARS:
        return Verdict(False, "too_long")
    if any(p.search(answer) for p in OUTPUT_LEAK):
        return Verdict(False, "meta_leak")
    if any(p.search(answer) for p in OUTPUT_CODE):
        return Verdict(False, "code")

    evidence = "\n".join(cited_texts)
    for link in _URL.findall(answer):
        if link.lower() not in evidence.lower():
            return Verdict(False, "unsourced_link")

    allowed = numbers_in(evidence) | {
        n for n in numbers_in(question) if _small(n)
    }
    ungrounded = numbers_in(answer) - allowed
    if ungrounded:
        return Verdict(False, "ungrounded_number")
    return ALLOW


def plain_punctuation(text: str) -> str:
    """Models love em dashes; customers' copy shouldn't have them. Ranges become "to",
    asides become commas. Numbers are untouched, so grounding checks still hold."""
    text = re.sub(r"(\d)\s*[\u2013\u2014]\s*(\d)", r"\1 to \2", text)
    text = re.sub(r"\s*[\u2013\u2014]\s*", ", ", text)
    return re.sub(r",\s*([.,!?])", r"\1", text).strip(" ,")


def _small(n: str) -> bool:
    try:
        return float(n.replace(":", ".")) <= MAX_ECHO_NUMBER
    except ValueError:
        return False
