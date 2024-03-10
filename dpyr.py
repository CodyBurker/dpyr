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


    def __or__(self, other):
        """
        Override the __or__ operator to allow for magrittr style piping
        """
            
        polars_df = other(self)
        return DataFrame(polars_df)

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
        if isinstance(other, DataFrame):
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
        return df.select(*self.args)


class filter(DataFrameOperation):
    """
    Filter rows from a DataFrame. This is equivalent to, and wrapper of, polars' filter method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | filter(c.column_1 > 1)
    ```
    """

    def __call__(self, df):
        """
        Apply the filter operation on the DataFrame
        """
        return df.filter(*self.args)

class mutate(DataFrameOperation):
    """
    Add or change a column in a DataFrame. This is equivalent to, and wrapper of, polars' with_column method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | mutate(new_column = c.column_1 + c.column_2)
    ```
    """

    def __call__(self, df):
        """
        Apply the mutate operation on the DataFrame
        """
        return df.with_columns(**self.kwargs)

# class group_by(DataFrameOperation):
#     """
#     Group a DataFrame
#     """

#     def __call__(self, df):
#         """
#         Apply the group_by operation on the DataFrame
#         """
#         print('Grouping by:', self.args)
#         print('original df:', df)
#         print('type:', type(df))
#         return df.group_by(*self.args)

# class summarize(DataFrameOperation):
#     """
#     Summarise a DataFrame, equivilent to dplyr's summarise and polars' agg
#     """

#     def __call__(self, df):
#         """
#         Apply the summarise operation on the DataFrame
#         """
#         return df.agg(**self.kwargs)

# if __name__ == '__main__':
#     # Create a dataframe
#     df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
#     # Test the groupby operation
#     df = df | group_by("a")