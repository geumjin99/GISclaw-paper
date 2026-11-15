"""Prompt texts per tier. H0 and H1 contain no GIS rules; all GIS guidance is component G5
(gis_rules.md), which tier H2 appends to the system prompt."""
import os

from protocol import tool_descriptions

HERE = os.path.dirname(os.path.abspath(__file__))

PACKAGES = ("geopandas, rasterio, shapely, fiona, pyproj, numpy, pandas, scipy, matplotlib, sklearn, "
            "libpysal, esda, mgwr, xarray, rasterstats, networkx, osmnx, seaborn, mapclassify, h3, momepy, "
            "pointpats, spaghetti, openpyxl, rtree, geoplot, cartopy, imblearn, pykrige, skimage")
NOT_AVAILABLE = "arcpy (use geopandas + rasterio + shapely instead)"

BASE = """You are an agent that completes data analysis tasks by writing and running Python code.

## Environment
- A persistent Python environment: variables persist between code executions. There is no internet access.
- Input data is in dataset/ (read-only). Save every deliverable in pred_results/.
- Available packages: {packages}. Not available: {not_available}.
- Figures: use matplotlib with plt.savefig(...); plt.show() does nothing.

## Tools
{tools}

{fmt}"""

GENERIC_GUIDANCE = """
## Working method
- Look at the data before computing: print sizes, column names and a few values.
- Work in small steps (a few to ~20 lines per execution) and print intermediate results to check them.
- Save each deliverable in the exact format and file name the task asks for, then list pred_results/ to confirm.
- Call plt.savefig in the same execution that draws the figure, then plt.close().
- Call finish only when every requested deliverable is saved."""

CALL_FMT = ("## Calling tools\nCall exactly one tool per reply. Explain your reasoning briefly in the message "
              "text, then make the call. The system returns the result.")


def gis_rules():
    with open(os.path.join(HERE, "gis_rules.md")) as f:
        return f.read().strip()


def system_prompt(tools, generic, g5):
    s = BASE.format(packages=PACKAGES, not_available=NOT_AVAILABLE, tools=tool_descriptions(tools), fmt=CALL_FMT)
    if generic:
        s += "\n" + GENERIC_GUIDANCE
    if g5:
        s += "\n\n" + gis_rules()
    return s


def task_message(task, file_list, data_card=None):
    s = f"Task: {task['instruction'].strip()}\n"
    if task.get("dataset_description"):
        s += f"\n## Dataset Description\n{task['dataset_description'].strip()}\n"
    s += f"\n## Files\n{file_list}\n"
    if task.get("domain_knowledge"):
        s += f"\n## Domain Knowledge\n{task['domain_knowledge'].strip()}\n"
    if task.get("hint"):
        s += f"\n## Additional Note\n{task['hint'].strip()}\n"
    if data_card:
        s += f"\n## Data profile (generated automatically by the system)\n{data_card}\n"
    return s
