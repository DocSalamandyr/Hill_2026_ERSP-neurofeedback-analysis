"""Group-level metadata and batch management — replaces EEGLAB STUDY.

Provides:
- Subject-to-group mapping with metadata (exclusion flags, session dates)
- File-path resolution for every pipeline intermediate
- Batch iteration helpers ("for each active subject, for each session…")
- Processing-state tracking (validated → preprocessed → epoched → ERSP)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .config import (
    ANALYSES_ROOT,
    BFB_SESSIONS,
    DATA_ROOT,
    DERIVATIVES_ROOT,
    GROUPS,
    GROUP_REWARD_BAND,
    GROUP_TRAINING_CHANNEL,
    RESTING_SESSIONS,
    RESTING_TYPES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Processing state
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    RAW = "raw"
    VALIDATED = "validated"
    PREPROCESSED = "preprocessed"
    EPOCHED = "epoched"
    ERSP_COMPUTED = "ersp_computed"
    ERP_COMPUTED = "erp_computed"


# ---------------------------------------------------------------------------
# Subject record
# ---------------------------------------------------------------------------


@dataclass
class Subject:
    subject_id: str
    group: str                              # key into config.GROUPS
    excluded: bool = False
    exclusion_reason: str = ""
    session_dates: Dict[int, str] = field(default_factory=dict)
    notes: str = ""
    stage: Stage = Stage.RAW
    known_bad_channels: List[str] = field(default_factory=list)
    interpolated_channels_original: List[str] = field(default_factory=list)

    @property
    def reward_band(self) -> str:
        return GROUP_REWARD_BAND[self.group]

    @property
    def training_channel(self) -> str:
        return GROUP_TRAINING_CHANNEL[self.group]

    @property
    def is_active(self) -> bool:
        return self.group != "sham"


# ---------------------------------------------------------------------------
# Study — the central registry
# ---------------------------------------------------------------------------


class Study:
    """Manages the full subject pool and derived file paths.

    Instantiate with a list of :class:`Subject` objects or load from a JSON
    file via :meth:`from_json`.
    """

    def __init__(
        self,
        subjects: Sequence[Subject],
        data_root: Path = DATA_ROOT,
        analyses_root: Path = ANALYSES_ROOT,
        derivatives_root: Path = DERIVATIVES_ROOT,
    ) -> None:
        self.data_root = data_root
        self.analyses_root = analyses_root
        self.derivatives_root = derivatives_root
        self._subjects: Dict[str, Subject] = {s.subject_id: s for s in subjects}

    # -- serialization -------------------------------------------------------

    def to_json(self, path: Path) -> None:
        """Persist study metadata to a JSON file."""
        records = []
        for s in self._subjects.values():
            records.append({
                "subject_id": s.subject_id,
                "group": s.group,
                "excluded": s.excluded,
                "exclusion_reason": s.exclusion_reason,
                "session_dates": s.session_dates,
                "notes": s.notes,
                "stage": s.stage.value,
                "known_bad_channels": s.known_bad_channels,
                "interpolated_channels_original": s.interpolated_channels_original,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2) + "\n")
        logger.info("Study metadata saved to %s", path)

    @classmethod
    def from_json(
        cls,
        path: Path,
        data_root: Path = DATA_ROOT,
        analyses_root: Path = ANALYSES_ROOT,
        derivatives_root: Path = DERIVATIVES_ROOT,
    ) -> "Study":
        """Load study metadata from a JSON file."""
        records = json.loads(path.read_text())
        subjects = []
        for r in records:
            subjects.append(Subject(
                subject_id=r["subject_id"],
                group=r["group"],
                excluded=r.get("excluded", False),
                exclusion_reason=r.get("exclusion_reason", ""),
                session_dates=r.get("session_dates", {}),
                notes=r.get("notes", ""),
                stage=Stage(r.get("stage", "raw")),
                known_bad_channels=r.get("known_bad_channels", []),
                interpolated_channels_original=r.get("interpolated_channels_original", []),
            ))
        return cls(subjects, data_root, analyses_root, derivatives_root)

    # -- accessors -----------------------------------------------------------

    def __getitem__(self, subject_id: str) -> Subject:
        return self._subjects[subject_id]

    def __len__(self) -> int:
        return len(self._subjects)

    def __iter__(self) -> Iterator[Subject]:
        return iter(self._subjects.values())

    @property
    def subject_ids(self) -> List[str]:
        return list(self._subjects.keys())

    def included(self) -> List[Subject]:
        """Return non-excluded subjects."""
        return [s for s in self if not s.excluded]

    def by_group(self, group: str) -> List[Subject]:
        return [s for s in self.included() if s.group == group]

    def active_subjects(self) -> List[Subject]:
        return [s for s in self.included() if s.is_active]

    def sham_subjects(self) -> List[Subject]:
        return self.by_group("sham")

    def set_stage(self, subject_id: str, stage: Stage) -> None:
        self._subjects[subject_id].stage = stage

    # -- path resolution -----------------------------------------------------

    def raw_bdf(self, subject_id: str, rec_type: str, session: int) -> Path:
        return (
            self.data_root / subject_id
            / f"{subject_id}_{rec_type}_{session}.bdf"
        )

    def _analysis(self, subject_id: str, stage: str) -> Path:
        """Intermediate working files (preprocessed, epochs)."""
        d = self.analyses_root / subject_id / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _deriv(self, subject_id: str, stage: str) -> Path:
        """Final pipeline outputs (ERSP, ERP, resting, figures)."""
        d = self.derivatives_root / subject_id / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def preprocessed_fif(self, subject_id: str, session: int) -> Path:
        return (
            self._analysis(subject_id, "preprocessed")
            / f"{subject_id}_BFB_{session}_clean-raw.fif"
        )

    def ica_fif(self, subject_id: str, session: int) -> Path:
        return (
            self._analysis(subject_id, "preprocessed")
            / f"{subject_id}_BFB_{session}-ica.fif"
        )

    def epochs_fif(self, subject_id: str, session: int) -> Path:
        return (
            self._analysis(subject_id, "epochs")
            / f"{subject_id}_BFB_{session}_reward-epo.fif"
        )

    def ersp_h5(self, subject_id: str, session: int) -> Path:
        return (
            self._deriv(subject_id, "ersp")
            / f"{subject_id}_BFB_{session}_ersp.h5"
        )

    def ersp_thirds_h5(self, subject_id: str, session: int, third: int) -> Path:
        return (
            self._deriv(subject_id, "ersp")
            / f"{subject_id}_BFB_{session}_third{third}-ersp.h5"
        )

    def erp_h5(self, subject_id: str, session: int) -> Path:
        return (
            self._deriv(subject_id, "erp")
            / f"{subject_id}_BFB_{session}_erp.h5"
        )

    def resting_h5(
        self, subject_id: str, resting_type: str, session: int,
    ) -> Path:
        return (
            self._deriv(subject_id, "resting")
            / f"{subject_id}_{resting_type}_{session}_psd.h5"
        )

    # -- batch iterators -----------------------------------------------------

    def iter_bfb(
        self,
        subjects: Optional[Sequence[str]] = None,
        sessions: Sequence[int] = BFB_SESSIONS,
    ) -> Iterator[Tuple[Subject, int]]:
        """Yield ``(Subject, session)`` for each BFB recording."""
        pool = (
            [self._subjects[sid] for sid in subjects]
            if subjects is not None
            else self.included()
        )
        for subj in pool:
            for sess in sessions:
                yield subj, sess

    def iter_resting(
        self,
        subjects: Optional[Sequence[str]] = None,
        sessions: Sequence[int] = RESTING_SESSIONS,
        resting_types: Sequence[str] = RESTING_TYPES,
    ) -> Iterator[Tuple[Subject, str, int]]:
        """Yield ``(Subject, resting_type, session)``."""
        pool = (
            [self._subjects[sid] for sid in subjects]
            if subjects is not None
            else self.included()
        )
        for subj in pool:
            for rtype in resting_types:
                for sess in sessions:
                    yield subj, rtype, sess

    # -- summary -------------------------------------------------------------

    def summary(self) -> str:
        lines = ["Study summary", "=" * 40]
        for gkey, glabel in GROUPS.items():
            members = self.by_group(gkey)
            ids = ", ".join(s.subject_id for s in members)
            lines.append(f"  {glabel}: n={len(members)}  [{ids}]")
        excluded = [s for s in self if s.excluded]
        if excluded:
            lines.append(f"  Excluded: {len(excluded)}")
        lines.append(f"  Total included: {len(self.included())}")
        return "\n".join(lines)
