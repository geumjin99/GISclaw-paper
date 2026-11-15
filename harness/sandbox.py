"""Persistent Python execution environment (runs inside the container, see kernel.py).

One namespace per task, as in a notebook: variables persist between executions. Common GIS
libraries are imported once at start. Each execution returns its stdout, stderr, success flag
and a short description of the variables it created.
"""
import io
import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict


class ExecutionResult:
    """Result of a single code execution."""
    def __init__(self, stdout: str = "", stderr: str = "",
                 success: bool = True, new_vars: Dict[str, str] = None,
                 execution_time: float = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.new_vars = new_vars or {}  # {var_name: type_summary}
        self.execution_time = execution_time


class PythonSandbox:
    """Interactive Python execution sandbox."""

    # Packages the agent is told are unavailable (checked before each execution);
    # the harness replaces this mapping with `blocked_packages` from the configuration.
    BLOCKED_PACKAGES = {
        'arcpy': 'Use geopandas + rasterio + shapely instead.',
    }

    def __init__(self, work_dir: str = ".", timeout: int = 60):
        self.work_dir = os.path.abspath(work_dir)
        self.timeout = timeout
        self.namespace = {"__builtins__": __builtins__}
        self.code_history = []  # successfully executed code blocks
        self._known_vars = set()
        self._init_environment()

    def _init_environment(self):
        """Import the common libraries and change to the working directory."""
        init_code = """
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.close('all')
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)
"""
        os.chdir(self.work_dir)

        try:
            exec(init_code, self.namespace)
            self._known_vars = set(self.namespace.keys())
        except Exception as e:
            print(f"Sandbox init warning: {e}")

    def execute(self, code: str) -> ExecutionResult:
        """Execute a code block in the persistent namespace."""
        import time
        import threading

        # refuse blocks that import a package listed as unavailable
        BLOCKED_PACKAGES = self.BLOCKED_PACKAGES
        for line in code.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            for pkg, alt in BLOCKED_PACKAGES.items():
                if f'import {pkg}' in stripped or f'from {pkg}' in stripped:
                    return ExecutionResult(
                        stdout="",
                        stderr=f"\u274c Package '{pkg}' is NOT available in this environment.\n"
                               f"Alternative: {alt}\n"
                               f"Do NOT attempt to import {pkg} again \u2014 use the alternative packages.",
                        success=False,
                        new_vars={},
                        execution_time=0.0,
                    )

        vars_before = set(self.namespace.keys())

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        t0 = time.time()
        success = True

        original_dir = os.getcwd()
        os.chdir(self.work_dir)

        timer = threading.Timer(self.timeout, lambda: None)
        timer.start()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace)
        except Exception:
            success = False
            stderr_buf.write(traceback.format_exc())
        finally:
            elapsed = time.time() - t0
            timer.cancel()
            os.chdir(original_dir)
            if elapsed > self.timeout:
                success = False
                stderr_buf.write(f"\n\u23f0 Execution took {elapsed:.1f}s, exceeding limit of {self.timeout}s")

        exec_time = time.time() - t0

        # variables created by this block
        vars_after = set(self.namespace.keys())
        new_var_names = vars_after - vars_before
        new_vars = {}
        for name in new_var_names:
            if name.startswith('_'):
                continue
            new_vars[name] = self._describe_var(name)

        result = ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            success=success,
            new_vars=new_vars,
            execution_time=exec_time,
        )

        if success:
            self.code_history.append(code)
            self._known_vars = vars_after

        return result

    def inspect(self, var_name: str) -> str:
        """Describe a variable in detail."""
        if var_name not in self.namespace:
            return f"\u274c Variable '{var_name}' does not exist. Available variables: {self._list_user_vars()}"

        return self._describe_var(var_name, detailed=True)

    def _describe_var(self, var_name: str, detailed: bool = False) -> str:
        """Short description of a variable: type, size, CRS, columns."""
        obj = self.namespace.get(var_name)
        if obj is None:
            return "None"

        try:
            import geopandas as gpd
            import pandas as pd
            import numpy as np

            if isinstance(obj, gpd.GeoDataFrame):
                desc = f"GeoDataFrame, {len(obj)} rows, CRS={obj.crs}"
                cols = [c for c in obj.columns if c != 'geometry']
                desc += f"\n  Columns: {', '.join(cols[:15])}"
                if detailed and len(obj) > 0:
                    desc += f"\n  First 3 rows:\n{obj.head(3).to_string()}"
                    for col in cols[:5]:
                        try:
                            uniques = obj[col].dropna().unique()
                            if len(uniques) <= 8:
                                desc += f"\n  {col} unique values: {list(uniques[:8])}"
                            else:
                                desc += f"\n  {col}: {len(uniques)} unique values, e.g.: {list(uniques[:3])}"
                        except:
                            pass
                return desc

            elif isinstance(obj, pd.DataFrame):
                desc = f"DataFrame, {len(obj)} rows, {len(obj.columns)} cols"
                desc += f"\n  Columns: {', '.join(obj.columns[:15])}"
                if detailed:
                    desc += f"\n  First 3 rows:\n{obj.head(3).to_string()}"
                return desc

            elif isinstance(obj, np.ndarray):
                desc = f"ndarray, shape={obj.shape}, dtype={obj.dtype}"
                if obj.size > 0:
                    desc += f", min={obj.min():.4g}, max={obj.max():.4g}"
                    if np.isnan(obj).any():
                        desc += f", NaN ratio={np.isnan(obj).mean():.1%}"
                return desc

            elif isinstance(obj, (gpd.GeoSeries,)):
                desc = f"GeoSeries, {len(obj)} geometries"
                if obj.crs:
                    desc += f", CRS={obj.crs}"
                return desc

            elif hasattr(obj, 'read'):  # an open rasterio dataset
                try:
                    desc = f"Raster, {obj.width}x{obj.height}, {obj.count} band(s)"
                    desc += f", CRS={obj.crs}, dtype={obj.dtypes[0]}"
                    if obj.nodata is not None:
                        desc += f", nodata={obj.nodata}"
                    return desc
                except:
                    pass

            s = str(obj)
            if len(s) > 200:
                s = s[:200] + "..."
            return f"{type(obj).__name__}: {s}"

        except Exception as e:
            return f"{type(obj).__name__} (description error: {e})"

    def _list_user_vars(self) -> str:
        """Names of the variables created by the agent (preloaded names and _private ones excluded)."""
        skip = {'__builtins__', 'os', 'np', 'pd', 'gpd', 'rasterio', 
                'matplotlib', 'plt', 'Point', 'Polygon', 'MultiPolygon',
                'box', 'unary_union', 'warnings', 'reproject',
                'calculate_default_transform'}
        user_vars = [k for k in self.namespace.keys() 
                     if not k.startswith('_') and k not in skip]
        return ', '.join(user_vars) if user_vars else "(none)"

    def get_full_code(self) -> str:
        """The successful blocks as one script, with the import header."""
        header = """import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)
"""
        return header + "\n\n" + "\n\n".join(self.code_history)
