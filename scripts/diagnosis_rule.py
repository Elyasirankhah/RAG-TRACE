import re

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
NAMES = {
    "NORM": ("normal",),
    "MI": ("myocardial infarction", "infarction", "infarct"),
    "STTC": ("st-t", "st/t", "st t", "sttc"),
    "CD": ("conduction disturbance", "conduction defect", "conduction block", "bundle branch"),
    "HYP": ("hypertrophy",),
}
BEFORE = re.compile(
    r"no evidence of|no clear evidence of|not consistent with|inconsistent with|"
    r"not possible|ruled out|rule out|no signs of|not confirmed|not supported|"
    r"not met|unlikely|less likely|less probable|cannot be excluded",
    re.I,
)
AFTER = re.compile(
    r"is not|not confirmed|not met|not supported|not possible|less likely|less probable|unlikely|cannot be excluded",
    re.I,
)
GLOBAL = re.compile(
    r"no diagnosis|cannot be made|no definitive|not a clinical diagnosis",
    re.I,
)


def _labels_from_field(raw) -> list[str]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = re.split(r"[,/]| and ", raw)
    else:
        return []
    found = []
    for value in values:
        name = str(value).strip().upper()
        if name in CLASSES and name not in found:
            found.append(name)
    return found


def negated(text: str) -> set[str]:
    low = text.lower()
    found = set()
    for label, names in NAMES.items():
        for name in names:
            for match in BEFORE.finditer(low):
                window = low[match.end(): match.end() + 70]
                if name in window:
                    found.add(label)
            start = 0
            while True:
                at = low.find(name, start)
                if at < 0:
                    break
                window = low[at + len(name): at + len(name) + 40]
                if AFTER.search(window):
                    found.add(label)
                start = at + len(name)
    if re.search(r"\bimpossible\b", low):
        for label, names in NAMES.items():
            if any(name in low for name in names):
                found.add(label)
    return found


def _from_text(text: str) -> list[str]:
    low = text.lower()
    found = []
    checks = (
        ("NORM", ("is normal", "interpreted as normal", "within normal", "normal sinus", "ecg is normal", "tracing is normal")),
        ("MI", ("myocardial infarction", "infarction", "infarct")),
        ("STTC", ("st-t", "st/t", "st t", "sttc")),
        ("CD", ("conduction disturbance", "conduction defect", "bundle branch")),
        ("HYP", ("hypertrophy",)),
    )
    for label, phrases in checks:
        if any(phrase in low for phrase in phrases):
            found.append(label)
    return found


def text_only_labels(text: str) -> list[str]:
    """Phrase extractor. It does not read the JSON label."""
    blocked = negated(text or "")
    return [label for label in _from_text(text or "") if label not in blocked]


def affirmed_labels(claim: dict) -> list[str]:
    if claim.get("type") != "diagnosis":
        return []
    text = str(claim.get("text") or "")
    if GLOBAL.search(text) and not _from_text(text):
        return []
    labels = _labels_from_field(claim.get("label"))
    if not labels:
        labels = _from_text(text)
    blocked = negated(text)
    return [label for label in labels if label not in blocked]
