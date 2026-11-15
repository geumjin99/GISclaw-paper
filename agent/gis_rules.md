## GIS analysis discipline

Follow these rules on every task, including ones that look simple.

### 1. Plan the whole analysis first
In your first Thought, write the full chain: inputs → operations → the exact files you must produce. Every step should feed a later step or a deliverable. If the task can be read in more than one way, pick the reading that yields a checkable result and say which one you chose.

### 2. Inspect the data before computing
Never assume a schema. For every input layer, print the feature count, geometry type, CRS and the actual column names; for rasters, the shape, band count, CRS, nodata value and value range. Print the range and the number of missing values of every field you are about to use. Take column names from the data, not from expectation.

### 3. Get the coordinate reference system right
- Check the CRS of every layer; do not assume they match.
- Distance, area, buffer and nearest-neighbour operations need a projected CRS in metres, not geographic degrees. Reproject first and state which CRS you chose and why.
- When layers disagree, reproject to the CRS of the layer that defines the reporting units.
- After a spatial join or overlay, print how many records matched. A join where almost nothing matched usually means a CRS or predicate error, not a property of the data.

### 4. Check the result before finishing
Files in `pred_results/` are not, by themselves, a correct result. Before `finish`, print and check:
- Counts: is the number of output features plausible? Zero rows, or a filter that removed nothing, signals a logic error.
- Ranges: are the values physically possible? An interpolated surface should stay close to the range of the observed values.
- Rasters: not all zero, all missing, or a single constant value.
- Maps and charts: the plotted field actually varies; a uniformly coloured map usually means the data behind it is wrong.
State the checked numbers in your Thought. If a check fails, fix the analysis rather than finishing.

### 5. Do not invent values for missing data
Units without a measurement stay missing. Do not fill them with a mean, zero or a neighbour's value to make a join, score or map look complete. If the computation truly cannot proceed without filling, say so first, report how many units were filled and how, and keep a column that marks them.

### 6. Diagnose errors instead of retrying
The real error is at the end of the traceback. Read it, form a hypothesis and change one thing. Do not re-run the same code, and do not hide an exception with a bare `try/except`.

### 7. Protect state and outputs
Do not overwrite a variable holding loaded input data; give derived data new names. Save every deliverable under `pred_results/` and list the folder to confirm. Call `plt.savefig(...)` in the same code block that draws the figure; never call `plt.show()`.
