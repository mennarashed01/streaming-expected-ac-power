from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .Timeutil import parse_timestamp

class Reading(BaseModel):
    # example shown (pre-filled) in /docs -> "Try it out"
    model_config = ConfigDict(json_schema_extra={"example": {
        "timestamp": "2022-06-21T18:00:00Z",
        "poa_irradiance": 900,
        "module_temp": 45,
        "ambient_temp": 25,
        "sensor_id": "PV10-01",
    }})

    timestamp: datetime = Field(..., description="ISO 8601 or e.g. 10/11/2010 9:18:00 AM (month first). No timezone = site local time (America/Denver); with Z/offset = that timezone")
    poa_irradiance: float = Field(..., ge=-50, le=1500, description="W/m2 (small negative night-time sensor noise is clipped to 0)")
    module_temp: float = Field(..., ge=-40, le=100, description="°C (MODULE_TEMP_2)")
    ambient_temp: float = Field(..., ge=-60, le=60, description="°C (AMBIENT_TEMP)")

    sensor_id: Optional[str] = Field(None, max_length=64, description="Optional: which sensor sent this reading (shown on the dashboard)")

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, v):
        return parse_timestamp(v)

    @field_validator("poa_irradiance")
    @classmethod
    def _clip_poa(cls, v: float) -> float:
        return max(v, 0.0)


class Prediction(BaseModel):
    timestamp: str
    expected_power_norm: float
    daytime: bool