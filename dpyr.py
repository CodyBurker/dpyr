import polars as pl

class DataFrame(pl.DataFrame):
    """
    Wrapper class for the polars DataFrame
    """

    def __init__(self, *args, **kwargs):
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

    def __init__(self, *args):
        self.args = args

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


class SelectOperation(DataFrameOperation):
    """
    Select columns from a DataFrame
    """

    def __call__(self, df):
        """
        Apply the select operation on the DataFrame
        """
        return df.select(*self.args)


class FilterOperation(DataFrameOperation):
    """
    Filter rows from a DataFrame
    """

    def __call__(self, df):
        """
        Apply the filter operation on the DataFrame
        """
        return df.filter(*self.args)