import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


WGSL_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Param1Entry:
    idx: int
    code: str
    name: str
    param1: float


def _parse_vec4_components(vec4_src: str) -> List[float]:
    # vec4_src example: "0.0, 0.3, -0.73, -0.23"
    parts = [p.strip() for p in vec4_src.split(",")]
    floats: List[float] = []
    for p in parts:
        m = WGSL_FLOAT_RE.search(p)
        if not m:
            raise ValueError(f"Could not parse float from vec4 component: {p!r}")
        floats.append(float(m.group(0)))
    return floats


def parse_shared_wgsl_param1(shared_wgsl_path: str) -> Dict[int, Param1Entry]:
    with open(shared_wgsl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries: Dict[int, Param1Entry] = {}

    # Matches: // 17 V - Valine
    header_re = re.compile(r"^\s*//\s*(\d+)\s+([A-Z])\s+-\s+(.+?)\s*$")

    i = 0
    while i < len(lines):
        m = header_re.match(lines[i])
        if not m:
            i += 1
            continue

        idx = int(m.group(1))
        code = m.group(2)
        name = m.group(3)

        # Scan forward to find the array<vec4...>(...) line for this entry.
        j = i + 1
        data_line: Optional[str] = None
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            if s.startswith("//"):
                # Next header reached without finding data for this one.
                break
            if "array<vec4<f32>,6>(" in s:
                data_line = s
                break
            j += 1

        if data_line is None:
            i += 1
            continue

        vec4s = re.findall(r"vec4<f32>\(([^)]*)\)", data_line)
        if len(vec4s) < 4:
            raise ValueError(f"Unexpected AMINO_DATA row format for idx={idx}: {data_line}")

        # parameter1 is d[3].w => 4th vec4, 4th component
        d3 = _parse_vec4_components(vec4s[3])
        if len(d3) != 4:
            raise ValueError(f"Unexpected vec4 component count for idx={idx}: {vec4s[3]}")

        entries[idx] = Param1Entry(idx=idx, code=code, name=name, param1=d3[3])
        i += 1

    return entries


SENSOR_NAME_TO_TYPE_ID = {
    "Alpha Sensor": 22,
    "Beta Sensor": 23,
    "Energy Sensor": 24,
    "Agent Alpha Sensor": 34,
    "Agent Beta Sensor": 35,
    "Trail Energy Sensor (alpha)": 37,
    "Trail Energy Sensor (beta)": 37,
    "Alpha Magnitude Sensor": 38,
    "Alpha Magnitude Sensor (var)": 39,
    "Beta Magnitude Sensor": 40,
    "Beta Magnitude Sensor (var)": 41,
}


def _sensor_kind_from_name(organ_name: str) -> Optional[str]:
    if organ_name in SENSOR_NAME_TO_TYPE_ID:
        return organ_name
    return None


def generate_table(shared_wgsl_path: str, organ_table_path: str) -> List[dict]:
    param1 = parse_shared_wgsl_param1(shared_wgsl_path)

    # Amino acids are 0..19; organs are 20..41
    amino_param1_by_code = {e.code: e.param1 for e in param1.values() if 0 <= e.idx <= 19}

    promoters = {
        "V_Valine": "V",
        "M_Methionine": "M",
        "H_Histidine": "H",
        "Q_Glutamine": "Q",
    }

    rows: List[dict] = []

    with open(organ_table_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            modifier_idx = int(r["Modifier"].strip())
            modifier_code = r["Modifier_AA"].strip()

            # Only keep promoters that produce sensors in the table.
            for promoter_col, promoter_code in promoters.items():
                organ_name = (r.get(promoter_col) or "").strip()
                sensor_kind = _sensor_kind_from_name(organ_name)
                if sensor_kind is None:
                    continue

                organ_type_id = SENSOR_NAME_TO_TYPE_ID[sensor_kind]

                promoter_p1 = amino_param1_by_code.get(promoter_code)
                modifier_p1 = amino_param1_by_code.get(modifier_code)

                # Only the env-dye sensors (directional + magnitude) actually use promoter+modifier param1
                # in the shader sampling helpers. The agent/trail/energy sensors use different logic.
                uses_param1_gain = organ_type_id in (22, 23, 38, 39, 40, 41)

                combined = None
                gain_abs = None
                polarity = None
                if uses_param1_gain and promoter_p1 is not None and modifier_p1 is not None:
                    combined = promoter_p1 + modifier_p1
                    gain_abs = abs(combined)
                    polarity = 1 if combined >= 0 else -1

                rows.append(
                    {
                        "sensor_kind": sensor_kind,
                        "organ_type_id": organ_type_id,
                        "promoter_code": promoter_code,
                        "promoter_param1": promoter_p1,
                        "modifier_index": modifier_idx,
                        "modifier_code": modifier_code,
                        "modifier_param1": modifier_p1,
                        "combined_param1": combined,
                        "gain_abs": gain_abs,
                        "polarity": polarity,
                        "notes": (
                            "env_dye_sensor (gain=abs(p+m), sign=sign(p+m))"
                            if uses_param1_gain
                            else "non_param1_sensor (gain not derived from promoter/modifier param1 in shader)"
                        ),
                    }
                )

    # Add the Energy Sensor (24) even though it isn’t in ORGAN_TABLE.csv.
    # It exists in the shader lookup table and can contribute to oscillations.
    if 24 in param1:
        rows.append(
            {
                "sensor_kind": "Energy Sensor",
                "organ_type_id": 24,
                "promoter_code": "",
                "promoter_param1": "",
                "modifier_index": "",
                "modifier_code": "",
                "modifier_param1": "",
                "combined_param1": "",
                "gain_abs": "",
                "polarity": "",
                "notes": "energy_sensor (uses energy->signal mapping; not promoter/modifier param1 gain)",
            }
        )

    # Sort for readability
    def sort_key(d: dict) -> Tuple:
        return (
            d["sensor_kind"],
            str(d["promoter_code"]),
            int(d["modifier_index"]) if isinstance(d["modifier_index"], int) else 999,
        )

    rows.sort(key=sort_key)
    return rows


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shared = os.path.join(repo_root, "shaders", "shared.wgsl")
    organ_table = os.path.join(repo_root, "config", "ORGAN_TABLE.csv")
    out_path = os.path.join(repo_root, "docs", "sensor_power_table.csv")

    rows = generate_table(shared, organ_table)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sensor_kind",
                "organ_type_id",
                "promoter_code",
                "promoter_param1",
                "modifier_index",
                "modifier_code",
                "modifier_param1",
                "combined_param1",
                "gain_abs",
                "polarity",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
