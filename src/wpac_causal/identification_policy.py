"""Deterministic backdoor adjustment-set policy for DoWhy 0.14 identification.

Why this module exists
----------------------
DoWhy 0.14 does not return a unique adjustment set.  Its default backdoor
identifier enumerates candidate sets by iterating over Python ``set`` objects
of eligible variable names (``itertools.combinations(filt_eligible_variables,
...)`` in ``dowhy.causal_identifier.auto_identifier``) and keeps the first
legal set of the smallest size.  String hashing is salted per process, so the
same DAG and the same estimation panel can yield different -- equally valid --
minimal backdoor sets under different ``PYTHONHASHSEED`` values.  Measured on
the HOLDOUT_MASKED_V1 prior panel (same identification DAG, same 12,647 rows):

    PYTHONHASHSEED   PV->DV24                 OHC_OA->DV24
    0                {SHRD,W500,WARM_CORE}    {LAT,SST_IC,VORT,W500}
    2                {SHRD,W500,WARM_CORE}    {SHRD,SST_OA,W500,WARM_CORE}
    3                {SHRD,SST_IC,W500}       {SHRD,SST_IC,SST_OA,W500}
    7                {SHRD,W500,WARM_CORE}    {PV,SHRD,SST_OA,W500}
    11               {SHRD,W500,WARM_CORE}    {SST_IC,SST_OA,VORT,W500}

Because the adjustment set changes, the PV/OHC_OA ATEs change between runs.
That is an identification-reproducibility defect, not a DAG error: every set
above satisfies the backdoor criterion for the same graph.

The deterministic rule
----------------------
1. Candidates are every distinct set stored in
   ``identified.backdoor_variables`` (DoWhy keeps all legal sets there:
   the maximal one first and then every legal set of the minimal size).  The
   ``"backdoor"`` entry is DoWhy's alias for its default pick and is dropped.
2. Each candidate is normalized: variables stripped, de-duplicated, sorted.
   Candidate order never matters (the whole set of candidates, not DoWhy's
   iteration order, determines the answer).
3. A candidate is illegal if it contains a column absent from the estimation
   panel, a forbidden column (treatment/outcome/id), or any NaN.  Illegal
   candidates are excluded and reported.
4. Among the legal candidates, one that strictly contains another legal
   candidate is the non-minimal "full" set and is dropped.
5. The selected set is ``min((len(candidate), candidate))`` -- fewest
   variables first, then lexicographic order of the sorted tuple.  The empty
   adjustment set is legal whenever DoWhy lists it (or when the outcome is not
   a descendant of the treatment, where DoWhy returns no candidates at all).
6. If no legal candidate remains the policy raises instead of falling back to
   an unadjusted fit, so the estimation sample (``n_obs``) can never shrink or
   grow silently.

The rule is ATE-blind: it never looks at effect sizes, p-values, signs or
significance.  It is a fixed tie-break among sets DoWhy already certified as
valid, nothing more.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

POLICY_ID = "deterministic_minimal_backdoor_lexicographic_v1"
DEFAULT_ALIAS_KEY = "backdoor"

STATUS_SELECTED = "selected"
STATUS_NOT_SELECTED = "not_selected"
STATUS_NON_MINIMAL = "excluded_non_minimal"
STATUS_MISSING_COLUMN = "excluded_missing_column"
STATUS_FORBIDDEN_COLUMN = "excluded_forbidden_column"
STATUS_MISSING_VALUES = "excluded_missing_values"
STATUS_NO_PATH = "excluded_no_directed_path"

POLICY_RULE = (
    "Candidates are every distinct set in identified.backdoor_variables "
    "(default 'backdoor' alias dropped; variables sorted and de-duplicated). "
    "Candidates containing a column absent from the estimation panel, a "
    "forbidden column, or any NaN are excluded. Among the remaining legal "
    "candidates, a set that strictly contains another legal candidate is "
    "non-minimal and excluded. The selection is min((len(candidate), sorted "
    "tuple)): fewest variables, then lexicographic. The empty set is legal. "
    "ATEs, p-values, signs and significance are never consulted; with no legal "
    "candidate the policy fails closed."
)


class IdentificationPolicyError(RuntimeError):
    """No legal adjustment set under :data:`POLICY_ID` (fail closed)."""


@dataclass(frozen=True)
class AdjustmentCandidate:
    """One candidate set plus why it was or was not selected."""

    variables: Tuple[str, ...]
    status: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"variables": list(self.variables), "status": self.status,
                "reason": self.reason}


@dataclass(frozen=True)
class AdjustmentSetDecision:
    """Auditable outcome of :data:`POLICY_ID` for one (treatment, outcome)."""

    treatment: str
    outcome: str
    selected: Tuple[str, ...]
    candidates: Tuple[AdjustmentCandidate, ...]
    no_directed_path: bool = False
    source: str = "dowhy_backdoor_sets"
    policy: str = POLICY_ID

    @property
    def rule(self) -> str:
        return POLICY_RULE

    def to_audit(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "rule": POLICY_RULE,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "source": self.source,
            "no_directed_path": self.no_directed_path,
            "selected": list(self.selected),
            "selection_key": [len(self.selected), list(self.selected)],
            "n_candidates": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def to_row_fields(self) -> Dict[str, Any]:
        """Columns an ATE/manifest row can carry as identification provenance."""
        return {
            "identification_policy": self.policy,
            "identification_source": self.source,
            "n_identification_candidates": len(self.candidates),
            "identification_candidates": json.dumps(
                [candidate.to_dict() for candidate in self.candidates],
                sort_keys=True),
        }


def normalize_candidate(variables: Iterable[Any]) -> Tuple[str, ...]:
    """Strip, de-duplicate and sort the variables of one candidate set."""
    unique: List[str] = []
    seen = set()
    for value in variables or ():
        if value is None:
            continue
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return tuple(sorted(unique))


def candidate_sets(identified: Any) -> List[Tuple[str, ...]]:
    """All distinct candidate sets DoWhy stored in ``identified``.

    ``identified.backdoor_variables`` is a dict in DoWhy 0.14 with numbered
    entries (every legal set DoWhy found) plus a ``"backdoor"`` alias for the
    default pick.  A bare list (older APIs / tests) is also accepted.
    """
    raw = getattr(identified, "backdoor_variables", None)
    if raw is None:
        return []
    if isinstance(raw, dict):
        numbered = [value for key, value in raw.items()
                    if str(key) != DEFAULT_ALIAS_KEY]
        entries = numbered if numbered else list(raw.values())
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = list(raw)
        scalar = all(not isinstance(item, (list, tuple, set, frozenset))
                     for item in items)
        entries = [items] if items and scalar else items
    else:
        entries = [raw]
    unique: List[Tuple[str, ...]] = []
    seen = set()
    for entry in entries:
        if entry is None:
            continue
        candidate = normalize_candidate(entry)
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return sorted(unique, key=lambda candidate: (len(candidate), candidate))


def _legality(df: pd.DataFrame, candidate: Tuple[str, ...],
              forbidden: Sequence[str]) -> Optional[Tuple[str, str]]:
    missing = [name for name in candidate if name not in df.columns]
    if missing:
        return STATUS_MISSING_COLUMN, "missing column(s): " + ", ".join(missing)
    forbidden_hits = [name for name in candidate if name in forbidden]
    if forbidden_hits:
        return (STATUS_FORBIDDEN_COLUMN,
                "forbidden column(s): " + ", ".join(forbidden_hits))
    with_nan = [name for name in candidate if bool(df[name].isna().any())]
    if with_nan:
        return (STATUS_MISSING_VALUES,
                "NaN in column(s): " + ", ".join(with_nan))
    return None


def decide_adjustment_set(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    candidates: Sequence[Sequence[str]],
    *,
    no_directed_path: bool = False,
    forbidden: Sequence[str] = (),
    source: str = "dowhy_backdoor_sets",
) -> AdjustmentSetDecision:
    """Apply :data:`POLICY_ID` to an explicit candidate collection."""
    normalized = sorted({normalize_candidate(candidate) for candidate in candidates},
                        key=lambda candidate: (len(candidate), candidate))
    forbidden = tuple(forbidden)
    if no_directed_path:
        audit = tuple(AdjustmentCandidate(candidate, STATUS_NO_PATH,
                                          "outcome is not a descendant of treatment")
                      for candidate in normalized)
        return AdjustmentSetDecision(
            treatment=treatment, outcome=outcome, selected=(),
            candidates=audit, no_directed_path=True, source="no_directed_path")
    if not normalized:
        raise IdentificationPolicyError(
            f"{treatment}->{outcome}: DoWhy returned no backdoor candidate set; "
            "refusing to fall back to an unadjusted fit")

    statuses: Dict[Tuple[str, ...], Tuple[str, str]] = {}
    legal: List[Tuple[str, ...]] = []
    for candidate in normalized:
        verdict = _legality(df, candidate, forbidden)
        if verdict is None:
            legal.append(candidate)
        else:
            statuses[candidate] = verdict

    if not legal:
        detail = "; ".join(
            f"{list(candidate)} {statuses[candidate][0]} "
            f"({statuses[candidate][1]})" for candidate in normalized)
        raise IdentificationPolicyError(
            f"{treatment}->{outcome}: no legal backdoor candidate set "
            f"(policy={POLICY_ID}); {detail}")

    legal_sets = {candidate: set(candidate) for candidate in legal}
    minimal = [candidate for candidate in legal
               if not any(other != candidate and other_set < legal_sets[candidate]
                          for other, other_set in legal_sets.items())]
    selected = min(minimal, key=lambda candidate: (len(candidate), candidate))

    audit = []
    for candidate in normalized:
        if candidate in statuses:
            status, reason = statuses[candidate]
        elif candidate == selected:
            status, reason = STATUS_SELECTED, "lexicographic minimum"
        elif candidate in minimal:
            status = STATUS_NOT_SELECTED
            reason = "legal minimal alternative; loses the deterministic tie-break"
        else:
            status = STATUS_NON_MINIMAL
            reason = "strictly contains another legal candidate"
        audit.append(AdjustmentCandidate(candidate, status, reason))

    return AdjustmentSetDecision(
        treatment=treatment, outcome=outcome, selected=selected,
        candidates=tuple(audit), source=source)


def decision_from_identified(
    df: pd.DataFrame,
    identified: Any,
    treatment: str,
    outcome: str,
    *,
    forbidden: Sequence[str] = (),
) -> AdjustmentSetDecision:
    """Apply the policy to a DoWhy ``IdentifiedEstimand``."""
    return decide_adjustment_set(
        df, treatment, outcome, candidate_sets(identified),
        no_directed_path=bool(getattr(identified, "no_directed_path", False)),
        forbidden=forbidden)


def identify_adjustment_set(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    graph_dot: str,
    *,
    forbidden: Sequence[str] = (),
) -> AdjustmentSetDecision:
    """Run DoWhy identification and return the deterministic decision.

    DoWhy is used for identification only; estimation stays in statsmodels.
    Raises whatever DoWhy raises (e.g. a broken/absent graph) or
    :class:`IdentificationPolicyError` when no candidate is legal.
    """
    from dowhy import CausalModel

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=graph_dot,
        proceed_when_unidentifiable=True,
    )
    identified = model.identify_effect()
    return decision_from_identified(df, identified, treatment, outcome,
                                    forbidden=forbidden)


def override_backdoor_variables(identified: Any,
                                variables: Sequence[str]) -> Any:
    """Pin DoWhy's estimand to a chosen adjustment set (for its refuters).

    Refuters read ``identified.get_adjustment_set()`` (i.e. the estimand's
    backdoor set), so leaving DoWhy's hash-dependent choice there would make
    the refutation verdicts -- and thus the trusted signs -- irreproducible
    even after the headline ATE is pinned.  Callers pass the policy-selected
    set (empty allowed) and get an estimand that reports it.
    """
    variables = list(variables)
    stored = getattr(identified, "backdoor_variables", None)
    if not isinstance(stored, dict):
        stored = {}
        identified.backdoor_variables = stored
    key = getattr(identified, "default_backdoor_id", None)
    if not key:
        key = DEFAULT_ALIAS_KEY
        identified.default_backdoor_id = key
    stored[key] = variables
    stored[DEFAULT_ALIAS_KEY] = variables
    return identified
