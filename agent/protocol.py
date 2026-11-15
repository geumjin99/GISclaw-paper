"""Action protocol: how the model tells the harness which tool to call.

The model uses the API's function calling; the tool calls come back structured and are
parsed here into one Action per reply. A reply that cannot be read as exactly one call of
a known tool with its arguments is a FormatError, which is reported back to the model.
Reply text is never executed as code.
"""
import ast
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Action:
    name: str
    args: dict
    thought: str = ""
    call_id: Optional[str] = None      # id of the tool call
    note: str = ""                     # e.g. "only the first of 2 calls was run"


@dataclass
class FormatError:
    message: str
    thought: str = ""
    kind: str = "format"               # empty | no_action | bad_args | unknown_tool
    detail: dict = field(default_factory=dict)


PROTOCOL = "native"                    # recorded with every run

# ---------------------------------------------------------------------------- tools
TOOL_SPECS = {
    "list_files": ("List the input files in dataset/ and the files you have saved in pred_results/, with type and size.", {}),
    "load_vector": ("Load a vector file (shp, geojson, gpkg, ...) into a GeoDataFrame named var_name "
                    "and show its size, CRS, geometry types, columns and a few rows.",
                    {"path": "file path, e.g. dataset/roads.shp", "var_name": "name of the variable to create"}),
    "load_raster": ("Load a raster file (tif, ...) into a float array named var_name (nodata replaced by NaN) "
                    "plus var_name_meta (crs, transform, nodata, ...), and show its size, CRS and value range.",
                    {"path": "file path, e.g. dataset/dem.tif", "var_name": "name of the variable to create"}),
    "inspect": ("Describe an existing variable: type, size, columns, CRS, value ranges.",
                {"var_name": "name of the variable"}),
    "execute": ("Run Python code in a persistent environment (variables persist across calls). "
                "Returns stdout, errors, warnings and new variables.",
                {"code": "Python code to run"}),
    "lookup_api": ("Search the installed GIS and spatial-analysis libraries (geopandas, rasterio, shapely, pyproj, "
                   "esda, libpysal, mgwr, pykrige, rasterstats, scipy, xarray, ...) and show the real signatures of "
                   "matching functions and classes. Use it instead of guessing an API.",
                   {"query": "what to look for, e.g. 'local getis ord', 'reproject raster', 'kriging'"}),
    "finish": ("Declare the task complete once all deliverables are saved in pred_results/.",
               {"summary": "short summary of what was produced"}),
}


def tool_descriptions(tools):
    lines = []
    for i, t in enumerate(tools, 1):
        desc, params = TOOL_SPECS[t]
        sig = ", ".join(params)
        lines.append(f"{i}. {t}({sig})\n   {desc}")
        for k, v in params.items():
            lines.append(f"   - {k}: {v}")
    return "\n".join(lines)


def tool_schema(tools):
    out = []
    for t in tools:
        desc, params = TOOL_SPECS[t]
        out.append({"type": "function", "function": {
            "name": t, "description": desc,
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                           "required": list(params)}}})
    return out


# ---------------------------------------------------------------------------- JSON
def _decode_object(s):
    """Decode the first JSON object in s. Returns (obj, end) or raises ValueError."""
    i = s.find("{")
    if i < 0:
        raise ValueError("no '{' found")
    dec = json.JSONDecoder(strict=False)  # literal newlines inside strings are common
    try:
        obj, end = dec.raw_decode(s[i:])
        return obj, i + end
    except json.JSONDecodeError as e:
        first_err = e
    # Python-literal style dict (single quotes, True/None): the balanced {...} span
    span = _balanced(s, i)
    if span:
        try:
            obj = ast.literal_eval(s[i:span])
            if isinstance(obj, dict):
                return obj, span
        except Exception:
            pass
    raise ValueError(f"invalid JSON: {first_err.msg} at char {first_err.pos}")


def _balanced(s, i):
    depth, q, esc = 0, None, False
    for j in range(i, len(s)):
        c = s[j]
        if q:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return None


# ---------------------------------------------------------------------------- parser
def parse_reply(content, tool_calls, tools):
    """tool_calls: list of {"id", "name", "arguments"(str)} from the API."""
    thought = (content or "").strip()
    if not tool_calls:
        if not thought:
            return FormatError("Your reply was empty. Call one of the tools.", kind="empty")
        return FormatError("Your reply did not call a tool. Call exactly one tool per reply "
                           "(use finish when the deliverables are saved).", thought=thought, kind="no_action")
    tc = tool_calls[0]
    note = "Only the first tool call was run; make one call per reply." if len(tool_calls) > 1 else ""
    name = tc.get("name")
    if name not in tools:
        return FormatError(f"Unknown tool '{name}'. Available tools: {', '.join(tools)}.", thought=thought,
                           kind="unknown_tool", detail={"name": name})
    raw = tc.get("arguments") or "{}"
    _, params = TOOL_SPECS[name]
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(args, dict):
            raise ValueError("not an object")
    except Exception as e:
        # a tool with ONE string parameter whose arguments came back as the bare value
        # (e.g. code instead of {"code": code}): the intent is unambiguous, so take the value
        # as that parameter (same for every tier and model)
        if len(params) == 1 and isinstance(raw, str) and raw.strip() and not raw.lstrip().startswith("{"):
            return Action(name, {next(iter(params)): raw}, thought, call_id=tc.get("id"),
                          note=note)
        return FormatError(f"The arguments of {name} are not valid JSON ({e}).", thought=thought, kind="bad_args")
    missing = [k for k in params if k not in args]
    if missing:
        return FormatError(f"{name} needs the argument(s) {', '.join(missing)}.", thought=thought, kind="bad_args")
    args = {k: (v if isinstance(v, str) else str(v)) for k, v in args.items() if k in params}
    return Action(name, args, thought, call_id=tc.get("id"), note=note)
