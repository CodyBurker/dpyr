"""Generate examples.ipynb, a full tour of every dpyr verb.

Run with: python build_notebook.py
Then it is executed in place so the outputs are saved.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(src):
    cells.append(new_code_cell(src))


md(
    "# dpyr — a full tour\n"
    "\n"
    "`dpyr` brings R's [dplyr](https://dplyr.tidyverse.org/) grammar of data "
    "manipulation to Python, on top of [polars](https://pola.rs/). You build a "
    "pipeline by piping a `DataFrame` through verbs with the `|` operator, and "
    "refer to columns symbolically with `c`.\n"
    "\n"
    "This notebook demonstrates every verb dpyr provides and notes where the "
    "behaviour matches dplyr.\n"
)

md("## Setup\n\nImport the verbs and a couple of helpers.")
code(
    "from dpyr import (\n"
    "    DataFrame, read_csv, c,\n"
    "    select, filter, mutate, arrange, desc,\n"
    "    head, tail, distinct, rename, count,\n"
    "    group_by, ungroup, summarize, summarise,\n"
    "    slice_sample, sample_n, pull,\n"
    "    inner_join, left_join, right_join, full_join, semi_join, anti_join,\n"
    "    n, lag, lead, row_number, min_rank, dense_rank,\n"
    ")\n"
    "import polars as pl"
)

md(
    "## The pipe and column references\n"
    "\n"
    "`c.column_name` is shorthand for `pl.col(\"column_name\")`. Use a double "
    "underscore for a column whose name contains a space (`c.my__col` ⇒ "
    "`pl.col(\"my col\")`), and dotted names in the source data become "
    "underscored (`sepal.length` ⇒ `sepal_length`)."
)
code(
    "df = read_csv(\"iris.csv\")\n"
    "df.head()"
)

md(
    "## `select` — pick columns\n"
    "Equivalent to dplyr's `select()`."
)
code("df | select(c.sepal_length, c.sepal_width, c.variety) | head()")

md("## `filter` — keep rows matching a condition")
code("df | filter(c.variety == \"Setosa\") | head()")

md(
    "## `mutate` — add or change columns\n"
    "Expressions are evaluated **sequentially**, just like dplyr, so a later "
    "expression can use a column created earlier in the same call."
)
code(
    "df | mutate(\n"
    "    sepal_ratio = c.sepal_length / c.sepal_width,\n"
    "    sepal_ratio_pct = c.sepal_ratio * 100,\n"
    ") | select(c.sepal_length, c.sepal_width, c.sepal_ratio, c.sepal_ratio_pct) | head()"
)

md(
    "## `arrange` — sort rows\n"
    "Wrap a column in `desc()` to sort descending. Like dplyr, missing values "
    "always sort to the end."
)
code("df | arrange(desc(c.sepal_length)) | head()")

md("## `head` / `tail` — first / last rows")
code("df | tail(3)")

md(
    "## `distinct` — unique rows\n"
    "With columns named, only those columns are kept (dplyr's default). Pass "
    "`keep_all=True` to keep every column."
)
code("df | distinct(c.variety)")

md("## `rename` — `rename(new = old)`")
code("df | rename(species = c.variety) | head()")

md(
    "## `count` — tally rows per group\n"
    "Returns the grouping columns plus a count column named `n`."
)
code("df | count(c.variety)")

md(
    "## `group_by` + `summarize` — split-apply-combine\n"
    "`group_by` produces a grouped frame; `summarize` aggregates within each "
    "group. `n()` gives the group size."
)
code(
    "df | group_by(c.variety) | summarize(\n"
    "    mean_sepal_length = c.sepal_length.mean(),\n"
    "    max_petal_length = c.petal_length.max(),\n"
    "    rows = n(),\n"
    ")"
)

md(
    "### Grouped `mutate` (window functions)\n"
    "Inside a grouped frame, `mutate` computes each expression **within** its "
    "group and keeps the grouping. Here every row gets its variety's mean sepal "
    "length, then we flag the above-average rows."
)
code(
    "df \\\n"
    "    | group_by(c.variety) \\\n"
    "    | mutate(variety_mean = c.sepal_length.mean()) \\\n"
    "    | filter(c.sepal_length > c.variety_mean) \\\n"
    "    | ungroup() \\\n"
    "    | select(c.variety, c.sepal_length, c.variety_mean) \\\n"
    "    | head()"
)

md(
    "### Ranking within groups\n"
    "`row_number()`, `min_rank()` and `dense_rank()` restart in each group."
)
code(
    "# row_number ranks ascending, so negate to rank the largest first.\n"
    "df \\\n"
    "    | group_by(c.variety) \\\n"
    "    | mutate(rank = row_number(-c.sepal_length)) \\\n"
    "    | filter(c.rank <= 2) \\\n"
    "    | ungroup() \\\n"
    "    | arrange(c.variety, c.rank) \\\n"
    "    | select(c.variety, c.sepal_length, c.rank)"
)

md(
    "## `lag` / `lead` — shifted values\n"
    "Like dplyr, `lag(x)[i] == x[i-1]` and the fill is null by default."
)
code(
    "df \\\n"
    "    | head(5) \\\n"
    "    | select(c.sepal_length) \\\n"
    "    | mutate(prev = lag(c.sepal_length), nxt = lead(c.sepal_length))"
)

md(
    "## Joins\n"
    "All six dplyr joins are available. Build two small tables to demonstrate."
)
code(
    "orders = DataFrame({\n"
    "    \"customer_id\": [1, 2, 1, 3, 2],\n"
    "    \"amount\": [100, 200, 50, 300, 75],\n"
    "})\n"
    "customers = DataFrame({\n"
    "    \"customer_id\": [1, 2, 4],\n"
    "    \"name\": [\"Alice\", \"Bob\", \"Dana\"],\n"
    "})\n"
    "orders"
)

md("### `inner_join` — only matching rows")
code("orders | inner_join(customers, by=c.customer_id)")

md("### `left_join` — keep all left rows")
code("orders | left_join(customers, by=c.customer_id)")

md("### `full_join` — keep all rows from both, coalescing the key (like dplyr)")
code("orders | full_join(customers, by=c.customer_id)")

md("### `semi_join` / `anti_join` — filter by presence in another table")
code(
    "# Orders whose customer has a profile vs. orders that don't.\n"
    "matched = orders | semi_join(customers, by=c.customer_id)\n"
    "orphans = orders | anti_join(customers, by=c.customer_id)\n"
    "display(matched)\n"
    "display(orphans)"
)

md(
    "### A realistic join pipeline\n"
    "Total spend per named customer."
)
code(
    "orders \\\n"
    "    | inner_join(customers, by=c.customer_id) \\\n"
    "    | group_by(c.name) \\\n"
    "    | summarize(total = c.amount.sum(), orders = n()) \\\n"
    "    | arrange(desc(c.total))"
)

md(
    "## `slice_sample` / `sample_n` — random rows\n"
    "Use `n=` for a fixed count or `prop=` for a fraction."
)
code("df | slice_sample(n=5, seed=42)")

md("## `pull` — extract a single column as a Series")
code("df | distinct(c.variety) | pull(c.variety)")

md(
    "## Putting it all together\n"
    "The canonical demo pipeline: Setosa flowers, sepal ratio, top 5."
)
code(
    "read_csv(\"iris.csv\") \\\n"
    "    | filter(c.variety == \"Setosa\") \\\n"
    "    | select(c.sepal_length, c.sepal_width) \\\n"
    "    | mutate(sepal_ratio = c.sepal_length / c.sepal_width) \\\n"
    "    | arrange(desc(c.sepal_ratio)) \\\n"
    "    | head(5)"
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
with open("examples.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote examples.ipynb with", len(cells), "cells")
