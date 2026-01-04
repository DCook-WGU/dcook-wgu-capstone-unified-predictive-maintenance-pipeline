from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import json


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Ledger:
    stage: str
    recipe_id: str
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        kind: str,  # This is my step or decision
        step: str,
        message: str,
        why: Optional[str] = None,
        consequence: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        logger=None,
    ) -> Dict[str, Any]:
        entry = {
            "ts_utc": _now_utc_iso(),
            "stage": self.stage,
            "recipe": self.recipe_id,
            "kind": kind,
            "step": step,
            "message": message,
            "why": why,
            "consequence": consequence,
            "data": data or {},
        }
        self.entries.append(entry)

        # Optional immediate logging 
        if logger is not None:
            logger.info("LEDGER | %s", entry)

        return entry

    def write_json(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.entries, indent=2))
        return out_path





# Example usage
# ledger.add(
#    kind="decision",
#    step="label",
#    message="Derived anomaly_flag from machine_status",
#    why="Binary flag required for split EDA/modeling; Pump has multi-state status",
#    consequence="RECOVERING/BROKEN treated as anomalous",
#    data={"counts": silver_df["anomaly_flag"].value_counts().to_dict()},
#    logger=logger
# )