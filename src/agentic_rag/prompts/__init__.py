"""Versioned prompt files with a loader (ADR-005).

Prompts live beside this module as ``{name}.v{N}.md`` and are rendered with
``string.Template`` (``${var}``) so literal braces in prompt text (e.g. JSON
output contracts) need no escaping. Evals and benchmark records reference
prompts by ``Prompt.id`` (``"synthesis.v1"``) so results stay attributable to
the exact prompt text that produced them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from string import Template

_FILENAME = re.compile(r"^(?P<name>[a-z0-9-]+)\.v(?P<version>\d+)\.md$")


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}.v{self.version}"

    def render(self, **variables: str) -> str:
        """Substitute ``${var}`` placeholders; raises KeyError on missing vars."""
        return Template(self.text).substitute(**variables)


def load_prompt(name: str, version: int | None = None) -> Prompt:
    """Load a prompt by name; latest version unless one is pinned."""
    package = resources.files(__package__)
    versions: dict[int, str] = {}
    for entry in package.iterdir():
        m = _FILENAME.match(entry.name)
        if m and m.group("name") == name:
            versions[int(m.group("version"))] = entry.name
    if not versions:
        raise FileNotFoundError(f"No prompt files found for {name!r}")
    chosen = version if version is not None else max(versions)
    if chosen not in versions:
        raise FileNotFoundError(
            f"Prompt {name!r} has no version {chosen} (have {sorted(versions)})"
        )
    text = (package / versions[chosen]).read_text(encoding="utf-8")
    return Prompt(name=name, version=chosen, text=text)
