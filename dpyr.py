import polars as pl

class DataFrame(pl.DataFrame):
    """
    Wrapper class for the polars DataFrame. This class is used to add the magrittr style piping to the DataFrame. Use this class as a drop-in replacement for `polars.DataFrame`. For example:
    ```python
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ```

    You can then use all native polars methods on the DataFrame, such as:

    ```python
    df.write_csv("test.csv")
    ```
    You can also now use pipes with the DataFrame, in conjunction with other new methods documented here. For example:
    ```python
    df = df | select(c.a, c.b)
    ```
    is the same as:
    ```python
    df = df.select("a", "b")
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If column names contain periods, replace them with underscores
        self.columns = [col.replace('.', '_') for col in self.columns]

    def __or__(self, other):
        """
        Override the __or__ operator to allow for magrittr style piping.

        The operation is applied to this DataFrame. Results are wrapped back into
        a dpyr ``DataFrame`` so piping can continue, while grouped frames and
        non-frame results (such as a ``Series`` from ``pull``) are passed through
        unchanged.
        """
        result = other(self)
        if isinstance(result, pl.DataFrame) and not isinstance(result, DataFrame):
            return DataFrame(result)
        return result


class GroupedDataFrame:
    """
    Represents a DataFrame that has been grouped with :class:`group_by`.

    A ``GroupedDataFrame`` holds the underlying dpyr ``DataFrame`` together with
    the columns it was grouped by. Subsequent operations such as
    :class:`summarize`, :class:`mutate`, :class:`filter` or :class:`count`
    respect the grouping. It is created for you by ``df | group_by(...)`` and is
    not usually instantiated directly.
    """

    def __init__(self, df, keys):
        self.df = df
        self.keys = list(keys)

    def __or__(self, other):
        """
        Apply the next operation, respecting the current grouping.
        """
        result = other(self)
        if isinstance(result, pl.DataFrame) and not isinstance(result, DataFrame):
            return DataFrame(result)
        return result


class column:
    """
    Wrapper class to make column references easier to write
    """
    def __init__(self):
        pass

    def __getattr__(self, name):
        """
        Get the column reference
        """
        # Replace double underscores with spaces as a shortcut for column names
        name = name.replace('__', ' ')
        return pl.col(name)

# THIS NEEDS TO BE MADE BETTER
c = column()


def _column_name(expr):
    """
    Return the output column name for a polars expression or a plain string.
    """
    if isinstance(expr, str):
        return expr
    return expr.meta.output_name()


class _Desc:
    """
    Marker returned by :func:`desc` so :class:`arrange` knows to sort a column
    in descending order.
    """
    def __init__(self, expr):
        self.expr = expr


def desc(expr):
    """
    Sort a column in descending order inside :class:`arrange`, mirroring dplyr's
    ``desc()``. For example:
    ```python
    df = df | arrange(desc(c.column_1))
    ```
    """
    return _Desc(expr)


def n():
    """
    Number of rows in the current group (or whole frame), mirroring dplyr's
    ``n()``. Use inside :class:`summarize` or :class:`mutate`:
    ```python
    df = df | group_by(c.g) | summarize(count = n())
    ```
    """
    return pl.len()


def lag(expr, k=1, default=None):
    """
    Shift values forward by ``k`` rows, mirroring dplyr's ``lag()``:
    ```python
    df = df | mutate(prev = lag(c.value))
    ```
    """
    return expr.shift(k, fill_value=default)


def lead(expr, k=1, default=None):
    """
    Shift values backward by ``k`` rows, mirroring dplyr's ``lead()``:
    ```python
    df = df | mutate(next = lead(c.value))
    ```
    """
    return expr.shift(-k, fill_value=default)


class DataFrameOperation:
    """
    Base class for DataFrame operations. Used internally to allow for the magrittr style piping. Subclasses should implement the __call__ method to apply the operation to the DataFrame.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, df):
        """
        Apply the operation on the DataFrame
        """
        raise NotImplementedError("Subclasses should implement this!")

    def __or__(self, other):
        """
        Add ability to chain operations with DataFrame
        """
        if isinstance(other, (DataFrame, GroupedDataFrame)):
            return self(other)


class select(DataFrameOperation):
    """
    Select columns from a DataFrame. This is equivalent to, and wrapper of, polars' select method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | select(c.column_1, c.column_2)
    ```
    """

    def __call__(self, df):
        """
        Apply the select operation on the DataFrame
        """
        if isinstance(df, GroupedDataFrame):
            df = df.df
        return df.select(*self.args)


class filter(DataFrameOperation):
    """
    Filter rows from a DataFrame. This is equivalent to, and wrapper of, polars' filter method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | filter(c.column_1 > 1)
    ```

    When applied to a grouped frame the predicate is evaluated within each group,
    mirroring dplyr's grouped ``filter``.
    """

    def __call__(self, df):
        """
        Apply the filter operation on the DataFrame
        """
        if isinstance(df, GroupedDataFrame):
            predicates = [p.over(df.keys) for p in self.args]
            return df.df.filter(*predicates)
        return df.filter(*self.args)

class mutate(DataFrameOperation):
    """
    Add or change a column in a DataFrame. This is equivalent to, and wrapper of, polars' with_columns method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | mutate(new_column = c.column_1 + c.column_2)
    ```

    When applied to a grouped frame each expression is computed within its group
    (using a window function), mirroring dplyr's grouped ``mutate``.
    """

    def __call__(self, df):
        """
        Apply the mutate operation on the DataFrame
        """
        if isinstance(df, GroupedDataFrame):
            exprs = {name: expr.over(df.keys) for name, expr in self.kwargs.items()}
            return df.df.with_columns(**exprs)
        return df.with_columns(**self.kwargs)

class arrange(DataFrameOperation):
    """
    Arrange the rows in a DataFrame. This is equivalent to, and wrapper of, polars' sort method, but expects a dataframe to be piped to it. Wrap a column in :func:`desc` to sort it in descending order. For example:
    ```python
    df = df | arrange(c.column_1, desc(c.column_2))
    ```
    """

    def __call__(self, df):
        """
        Apply the arrange operation on the DataFrame
        """
        if isinstance(df, GroupedDataFrame):
            df = df.df
        exprs = []
        descending = []
        for arg in self.args:
            if isinstance(arg, _Desc):
                exprs.append(arg.expr)
                descending.append(True)
            else:
                exprs.append(arg)
                descending.append(False)
        return df.sort(exprs, descending=descending)

class head(DataFrameOperation):
    """
    Get the first n rows of a DataFrame. This is equivalent to, and wrapper of, polars' head method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | head(5)
    ```
    """

    def __call__(self, df):
        """
        Apply the head operation on the DataFrame
        """
        if isinstance(df, GroupedDataFrame):
            df = df.df
        return df.head(*self.args)


class tail(DataFrameOperation):
    """
    Get the last n rows of a DataFrame, mirroring dplyr's ``tail`` and wrapping
    polars' ``tail`` method. For example:
    ```python
    df = df | tail(5)
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        return df.tail(*self.args)


class group_by(DataFrameOperation):
    """
    Group a DataFrame by one or more columns, mirroring dplyr's ``group_by``.
    The result is a :class:`GroupedDataFrame` that subsequent verbs such as
    :class:`summarize`, :class:`mutate` and :class:`count` operate on per group.
    For example:
    ```python
    df = df | group_by(c.column_1) | summarize(total = c.column_2.sum())
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        keys = [_column_name(arg) for arg in self.args]
        return GroupedDataFrame(df, keys)


class ungroup(DataFrameOperation):
    """
    Remove grouping from a :class:`GroupedDataFrame`, mirroring dplyr's
    ``ungroup``. For example:
    ```python
    df = df | group_by(c.g) | mutate(...) | ungroup()
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            return df.df
        return df


class summarize(DataFrameOperation):
    """
    Summarise a DataFrame, equivalent to dplyr's ``summarise``/``summarize`` and
    polars' ``agg``. When applied to a :class:`GroupedDataFrame` it aggregates
    within each group; applied to an ungrouped frame it collapses to a single
    row. For example:
    ```python
    df = df | group_by(c.g) | summarize(total = c.value.sum(), count = n())
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            return df.df.group_by(df.keys, maintain_order=True).agg(**self.kwargs)
        return df.select(**self.kwargs)


# British spelling alias, as dplyr provides both.
summarise = summarize


class distinct(DataFrameOperation):
    """
    Keep only unique rows, mirroring dplyr's ``distinct``. With no arguments all
    columns are considered. When columns are supplied, rows are de-duplicated on
    those columns and only those columns are kept (dplyr's default); pass
    ``keep_all=True`` to retain every column. For example:
    ```python
    df = df | distinct(c.column_1)
    df = df | distinct(c.column_1, keep_all=True)
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        keep_all = self.kwargs.get("keep_all", False)
        if not self.args:
            return df.unique(maintain_order=True)
        subset = [_column_name(arg) for arg in self.args]
        deduped = df.unique(subset=subset, maintain_order=True)
        if keep_all:
            return deduped
        return deduped.select(subset)


class rename(DataFrameOperation):
    """
    Rename columns, mirroring dplyr's ``rename(new = old)``. For example:
    ```python
    df = df | rename(new_name = c.old_name)
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        mapping = {_column_name(expr): new for new, expr in self.kwargs.items()}
        return df.rename(mapping)


class count(DataFrameOperation):
    """
    Count the number of rows per unique combination of the given columns,
    mirroring dplyr's ``count``. The count is returned in a column named ``n``.
    With no columns the total number of rows is returned. For example:
    ```python
    df = df | count(c.column_1)
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            keys = df.keys + [_column_name(arg) for arg in self.args]
            df = df.df
        else:
            keys = [_column_name(arg) for arg in self.args]
        if not keys:
            return df.select(pl.len().alias("n"))
        return df.group_by(keys, maintain_order=True).agg(pl.len().alias("n"))


class slice_sample(DataFrameOperation):
    """
    Randomly sample rows, mirroring dplyr's ``slice_sample``/``sample_n``.
    Provide ``n`` for a fixed number of rows or ``prop`` for a fraction. For
    example:
    ```python
    df = df | slice_sample(n=5)
    df = df | slice_sample(prop=0.1)
    ```
    """

    def __init__(self, n=None, prop=None, replace=False, seed=None):
        self.n = n
        self.prop = prop
        self.replace = replace
        self.seed = seed

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        return df.sample(
            n=self.n,
            fraction=self.prop,
            with_replacement=self.replace,
            seed=self.seed,
        )


# dplyr also exposes sample_n; expose it as an alias.
sample_n = slice_sample


class pull(DataFrameOperation):
    """
    Extract a single column as a polars ``Series``, mirroring dplyr's ``pull``.
    For example:
    ```python
    values = df | pull(c.column_1)
    ```
    """

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        return df.get_column(_column_name(self.args[0]))


def _normalize_by(by):
    """
    Turn a ``by`` argument into the ``on`` value polars expects, or ``None`` when
    join keys should be inferred from common columns.
    """
    if by is None:
        return None
    if isinstance(by, (list, tuple)):
        return [_column_name(b) for b in by]
    return _column_name(by)


class _Join(DataFrameOperation):
    """
    Base class for the dplyr join verbs. Holds the right-hand DataFrame and the
    join keys.
    """

    how = "inner"

    def __init__(self, right, by=None, suffix="_right"):
        self.right = right
        self.by = by
        self.suffix = suffix

    def __call__(self, df):
        if isinstance(df, GroupedDataFrame):
            df = df.df
        on = _normalize_by(self.by)
        if on is None:
            return df.join(self.right, how=self.how, suffix=self.suffix)
        return df.join(self.right, on=on, how=self.how, suffix=self.suffix)


class inner_join(_Join):
    """
    Keep rows present in both frames, mirroring dplyr's ``inner_join``:
    ```python
    df = df | inner_join(other, by=c.id)
    ```
    """
    how = "inner"


class left_join(_Join):
    """
    Keep all rows from the left frame, mirroring dplyr's ``left_join``:
    ```python
    df = df | left_join(other, by=c.id)
    ```
    """
    how = "left"


class right_join(_Join):
    """
    Keep all rows from the right frame, mirroring dplyr's ``right_join``:
    ```python
    df = df | right_join(other, by=c.id)
    ```
    """
    how = "right"


class full_join(_Join):
    """
    Keep all rows from both frames, mirroring dplyr's ``full_join``:
    ```python
    df = df | full_join(other, by=c.id)
    ```
    """
    how = "full"


class semi_join(_Join):
    """
    Keep rows of the left frame that have a match in the right, mirroring dplyr's
    ``semi_join``:
    ```python
    df = df | semi_join(other, by=c.id)
    ```
    """
    how = "semi"


class anti_join(_Join):
    """
    Keep rows of the left frame that have no match in the right, mirroring
    dplyr's ``anti_join``:
    ```python
    df = df | anti_join(other, by=c.id)
    ```
    """
    how = "anti"


def read_csv(*args, **kwargs):
    """
    Reads a csv file into a DataFrame. This is equivalent to, and wrapper of, polars' read_csv method. Returns a dpyr DataFrame. For example:
    ```python
    df = read_csv("test.csv")
    ```
    """
    pl_df = pl.read_csv(*args, **kwargs)
    return DataFrame(pl_df)

class preview(DataFrameOperation):
    """
    Get a preview of the first n rows of a DataFrame, and display it in a notebook, and pass dataframe on to next operation or assignment.
    ```python
    df = df \
        | preview("Data preview") \
        | select(c.column_1, c.column_2) \
        | preview("Data preview 2")

    # df will be the same as it would without the preview operations:
    df2 = df | select(c.column_1, c.column_2)
    ```
    """
    def __init__(self, label: str,n=5):
        self.n = n
        self.label = label

    def __call__(self, df):
        """
        Apply the preview operation on the DataFrame
        """
        if isinstance(df, GroupedDataFrame):
            df = df.df
        if self.label:
            print(self.label)
        display(df.head(self.n))
        return df
