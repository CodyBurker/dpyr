# DPYR
A wrapper to introduce dplyr like syntax for data manipulation to pandas and polars.

Documentation [here](https://html-preview.github.io/?url=https://github.com/CodyBurker/dpyr/blob/main/html/dpyr.html)

Example notebooks: [demo.ipynb](https://github.com/CodyBurker/dpyr/blob/main/demo.ipynb) (pandas/polars/dpyr comparison) and [examples.ipynb](https://github.com/CodyBurker/dpyr/blob/main/examples.ipynb) (a full tour of every verb)

Goals: 
* Symbolic names for columns
    - `df | select(x, y)`
    - Detect collisions (e.g. 2 dataframes with the same name, 2 different columns with the same name)
* magittr like piping ability
    - `df | filter(x > 0) >> select(x, y)`
* dplyr verbs (mutate, select, filter, etc.)
    - `df | mutate(z = x + y)`
* A non-changing preview of the dataframe
    - `df2 = df | head(5) | select(x, y)` will preview `df.head(5)` and `df2 = df | select(x, y)`

## Backlog:
### Verbs
- [x] `select`
- [x] `filter`
- [x] `mutate`
- [x] `group_by`
- [x] `summarize`
- [x] `arrange`
- [x] `distinct`
- [x] `rename`
- [x] `count`
- [x] `head`
- [x] `tail`
- [x] `sample_n` (`slice_sample`)
- [x] `join` (`inner_join`, `left_join`, `right_join`, `full_join`, `semi_join`, `anti_join`)
- [x] Windows functions (`over` via grouped `mutate`, `lag`, `lead`)
- [x] `ungroup`
- [x] `pull`
### Symbolic column names
- [x] Initialize column names as variables

### Misc.
- [ ] Publish to PyPi
- [ ] Better documentation
- [x] Add tests
- [ ] Add CI/CD

## dplyr compatibility

dpyr aims to faithfully reproduce dplyr's semantics, not just polars'. In
particular:

* **`mutate` evaluates sequentially** — a later expression can reference a
  column created earlier in the same call.
* **`arrange` always sorts missing values last**, even under `desc()`.
* **Grouped `mutate`/`filter` preserve grouping**; `summarize` peels off the
  last grouping level (dplyr's `.groups = "drop_last"`).
* **`distinct(col)` keeps only the named columns** by default (`keep_all=True`
  keeps all), preserving row order.
* **`full_join` coalesces the join key** into a single column, like dplyr.
* **`count` on a grouped frame** adds to the existing grouping for the tally and
  restores the original grouping on the result.
* Window helpers `lag`, `lead`, `row_number`, `min_rank`, `dense_rank` and `n()`
  operate per group inside a grouped `mutate`.

Generate docs:
```python pdoc --html dpyr --force```
Unit and integration tests:
```python -m unittest tests test_integration```
Regenerate the example notebook:
```python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace examples.ipynb```