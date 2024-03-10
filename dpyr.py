import polars as pl

class DataFrame(pl.DataFrame):
    """
    Wrapper class for the polars DataFrame
    """

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], pl.dataframe.group_by.GroupBy):
            self.grouped = True
            self.group_by = args[0]
        else:
            super().__init__(*args, **kwargs)

    def __or__(self, other):
        """
        Override the __or__ operator to allow for magrittr style piping
        """
            
        polars_df = other(self)
        return DataFrame(polars_df)

class DataFrameOperation:
    """
    Base class for DataFrame operations
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
    Select columns from a DataFrame
    """

    def __call__(self, df):
        """
        Apply the select operation on the DataFrame
        """
        return df.select(*self.args)


class filter(DataFrameOperation):
    """
    Filter rows from a DataFrame
    """

    def __call__(self, df):
        """
        Apply the filter operation on the DataFrame
        """
        return df.filter(*self.args)

class mutate(DataFrameOperation):
    """
    Mutate a DataFrame
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