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
- [ ] `mutate` - this one might be difficult
- [ ] `summarize`
- [ ] `group_by`
- [ ] `arrange`
- [ ] `distinct`
- [ ] `rename`
- [ ] `count`

### 