# DPYR
A wrapper to introduce dplyr like syntax for data manipulation to pandas and polars.
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
- [ ] `group_by`
- [ ] `summarize`
- [ ] `arrange` - Need to add descending order, preferably via a `desc` function or - operator
- [ ] `distinct`
- [ ] `rename`
- [ ] `count`
- [x] `head`
- [ ] `tail`
- [ ] `sample_n`
- [ ] `join`
- [ ] Windows functions? Over, lag, lead, etc.
### Symbolic column names
- [x] Initialize column names as variables

### Misc.
- [ ] Publish to PyPi
- [ ] Better documentation
- [x] Add tests
- [ ] Add CI/CD

Generate docs:
```python pdoc --html dpyr --force```