"""Derive reliable time-sensitive facts from any dynamically injected JSON context."""

import json
import re
from datetime import datetime


_TIMESTAMP_PATTERN = re.compile(r"\[([0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2})\]")
_TIME_RANGE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*(?:h|:)([0-5]\d)?\s*-\s*(\d{1,2})\s*(?:h|:)([0-5]\d)?"
)
_SINGLE_TIME_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*(?:h|:)([0-5]\d)?(?!\d)")
_LABEL_FIELDS = ("titre", "title", "nom", "name", "label", "description")


def _timestamp_from_question(question):
    match = _TIMESTAMP_PATTERN.search(question or "")
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")


def _time_to_minutes(hour, minute=0):
    return int(hour) * 60 + int(minute or 0)


def _parse_time_interval(value):
    if not isinstance(value, str):
        return None
    match = _TIME_RANGE_PATTERN.search(value)
    if match:
        return _time_to_minutes(match.group(1), match.group(2)), _time_to_minutes(match.group(3), match.group(4))
    match = _SINGLE_TIME_PATTERN.search(value)
    if match:
        minute = _time_to_minutes(match.group(1), match.group(2))
        return minute, None
    return None


def _iter_context_sections(context):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\[([^\]]+)\]\s*$", context or ""):
        try:
            data, _ = decoder.raw_decode(context[match.end():].lstrip())
        except (json.JSONDecodeError, ValueError):
            continue
        yield match.group(1), data


def _item_label(item, path):
    if isinstance(item, dict):
        for field in _LABEL_FIELDS:
            value = item.get(field)
            if isinstance(value, str):
                return value
    return ".".join(path) or "element sans nom"


def _temporal_values(value, path=()):
    if isinstance(value, dict):
        for key, nested_value in value.items():
            yield from _temporal_values(nested_value, path + (str(key),))
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            yield from _temporal_values(nested_value, path + (str(index),))
    elif isinstance(value, str):
        interval = _parse_time_interval(value)
        if interval:
            yield path, value, interval


def _temporal_facts(section, data, current_time):
    now_minutes = _time_to_minutes(current_time.hour, current_time.minute)
    facts = []
    for path, value, (start, end) in _temporal_values(data):
        parent = data
        for segment in path[:-1]:
            parent = parent[int(segment)] if isinstance(parent, list) else parent[segment]
        label = _item_label(parent, path[:-1])
        field = path[-1]
        if end is None:
            state = "passee" if start < now_minutes else "a venir" if start > now_minutes else "maintenant"
        elif start <= end:
            state = "terminee" if end <= now_minutes else "en cours" if start <= now_minutes else "a venir"
        else:
            state = "terminee" if end <= now_minutes < start else "en cours"
        facts.append(f"[{section}] {label} - {field}: {value} ({state})")
    return facts


def build_temporal_context(question, context):
    """Return computed, schema-independent temporal facts for the current user turn."""
    current_time = _timestamp_from_question(question)
    if current_time is None:
        return ""

    facts = [
        "FACTS TEMPORELS CALCULES : utilise-les comme source de verite pour toute reponse dependant de l'heure.",
        f"Heure actuelle de reference : {current_time.strftime('%Y-%m-%d %H:%M')}.",
    ]
    temporal_facts = []
    for section, data in _iter_context_sections(context):
        temporal_facts.extend(_temporal_facts(section, data, current_time))

    facts.append("Elements temporels extraits : " + ("; ".join(temporal_facts) if temporal_facts else "aucun."))
    facts.append(
        "Pour toute reponse dependant de l'heure, cite uniquement des noms, horaires et details presents mot pour mot dans le contexte fourni ou dans les faits calcules. "
        "N'invente aucune entite, aucun horaire ni aucun statut ; si les faits ne permettent pas de repondre, dis que le contexte ne permet pas de le determiner."
    )
    return "\n".join(facts)