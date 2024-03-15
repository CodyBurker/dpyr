import polars as pl
import re

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

class arrange(DataFrameOperation):
    """
    Arrange the rows in a DataFrame. This is equivalent to, and wrapper of, polars' sort method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | arrange(c.column_1, c.column_2)
    ```
    """

    def __call__(self, df):
        """
        Apply the arrange operation on the DataFrame
        """
        return df.sort(*self.args, *self.kwargs)

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
        return df.head(*self.args)

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

    ``` 
    df2 = df
         | preview("Data preview")
         | select(c.column_1, c.column_2)
         | preview("Data preview 2")
    ```
    This code is the same as:
    ```
    print('Data preview')
    display(df.head(5))
    df2 = df.select("column_1", "column_2")
    print('Data preview 2')
    display(df2.head(5))
    ```
    The intent is to allow for previewing data in a notebook while developing a pipeline of operations.
    """
    def __init__(self, label: str,n=5):
        self.n = n
        self.label = label

    def __call__(self, df):
        """
        Apply the preview operation on the DataFrame
        """
        if self.label:
            print(self.label)
        display(df.head(self.n))
        return df

class distinct(DataFrameOperation):
    """
    Get the distinct rows of a DataFrame. This is equivalent to, and wrapper of, polars' unique method, but expects a dataframe to be piped to it. For example:
    ```python
    df = df | distinct()
    ```

    ```python
    df = df | distinct(c.column_1, c.column_2)
    ```
    """

    def __call__(self, df):
        """
        Apply the distinct operation on the DataFrame
        """
        # Loop over self.args to confirm all are strings.
        # If not, raise an error.
        parsed_args = []
        for arg in self.args:
            if not (isinstance(arg, str) or isinstance(arg, pl.expr.expr.Expr)):
                raise ValueError('Expected column but got expression')
            if isinstance(arg, pl.expr.expr.Expr):
                arg = extract_column_name_from_expr(arg)
            parsed_args.append(arg)
        self.args = parsed_args
        return df.unique(*self.args, *self.kwargs)

def extract_column_name_from_expr(expr):
    """
    Extracts the column name from a polars expression. For example:
    ```python
    extract_column_name_from_expr(c.column_1)
    # 'column_1'
    ```
    """
    if isinstance(expr, str):
        return expr
    # Check to make sure input is a polars expression
    if not isinstance(expr, pl.expr.expr.Expr):
        raise ValueError('Expected polars expression')
    # Get all referenced columns
    all_cols = re.findall(r'col\("(.+?)"\)', str(expr))
    # Check to see if there is more than one column referenced
    if len(all_cols) > 1:
        raise ValueError('Expeted column but got expression')
    # Return the column name
    return all_cols[0]

class rename(DataFrameOperation):
    """
    Rename columns in a DataFrame. For example:
    ```python
    print(df.columns)
    # ['column_1', 'column_2']
    df = df | rename(new_column = c.column_1, new_column_2 = "column_2")
    print(df.columns)
    # ['new_column', 'new_column_2']
    ```
    """

    def __call__(self, df):
        """
        Apply the rename operation on the DataFrame
        """
        # Get current column names
        cols = df.columns
        # Ensure no args have been passed (only kwargs allowed)
        if self.args:
            raise ValueError('Only keyword arguments allowed')
        # Ensure that all kwargs are valid column names
        for k,v in self.kwargs.items():
            v_parsed = extract_column_name_from_expr(v)
            if v_parsed not in cols:
                raise ValueError(f'Column {v_parsed} does not exist')
        current_col_names = {k: k for k in cols}
        # Renamed columns
        renamed_cols = {k: v for k, v in self.kwargs.items()}
        # Conver to strings if necessary
        renamed_cols = {old: extract_column_name_from_expr(new) for old,new in renamed_cols.items()}
        # Replace any values in current_col_names with the new column name if it exists
        for new, old in renamed_cols.items():
            if old in current_col_names:
                current_col_names[old] = new
        # Create the selector
        selector = [pl.col(old).alias(new) for old, new in current_col_names.items()]
        # Apply the selector to the DataFrame
        return df.select(*selector)


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