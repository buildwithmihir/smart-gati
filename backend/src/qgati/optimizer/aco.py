"""REMOVED — this module held the ACO (ant colony optimization) solver.

It is a tombstone, not a stub. Nothing imports it, nothing registers it, and
:data:`qgati.optimizer.registry.SOLVERS` no longer contains an ``aco`` entry, so
the project runs five solvers rather than six.

Why it is still on disk
-----------------------
Removing a file needs a shell, and this module was deleted on request rather than
by a tool that could unlink it. It is left as this notice instead of as working
code so that the removal is unambiguous: a re-added registry entry would fail
loudly on import rather than quietly bringing a sixth solver back into the
comparison, and a reader who greps for ``run_aco`` finds an explanation rather
than an implementation.

**Delete this file.** ``git rm src/qgati/optimizer/aco.py`` — the implementation
is in git history if it is ever wanted back.

Why the removal is worth recording
----------------------------------
ACO was not a redundant solver. When it was benchmarked it reached *lower raw
cost than QPSO* at n=15 and n=25, so its removal deletes the strongest published
counter-example to the project's headline solver. The two README tables that still
carry its figures are kept as measurements rather than re-run, and
``DESIGN_DECISIONS.md`` has a "Removed: ACO" section recording the finding. See
that section before citing any "QPSO performs best" claim from this repository.
"""

from __future__ import annotations

__all__: list[str] = []
