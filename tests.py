import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
from dpyr import (
    DataFrame,
    GroupedDataFrame,
    Columns,
    _sanitize_col_name,
    filter,
    select,
    mutate,
    c,
    arrange,
    desc,
    head,
    tail,
    read_csv,
    group_by,
    ungroup,
    summarize,
    summarise,
    distinct,
    rename,
    count,
    slice_sample,
    sample_n,
    pull,
    inner_join,
    left_join,
    right_join,
    full_join,
    semi_join,
    anti_join,
    n,
    lag,
    lead,
    row_number,
    min_rank,
    dense_rank,
)
import polars as pl

def compare_dpyr_polars(dpyr_result, polars_result):
    pandas_dpyr = dpyr_result.to_pandas()
    pandas_polars = polars_result.to_pandas()
    pandas_polars.columns = pandas_dpyr.columns
    assert_frame_equal(pandas_dpyr, pandas_polars)

class TestDpyr(unittest.TestCase):
    def setUp(self):
        base_df = {"a": [1, 2, 3], "b": [6, 5, 4],"test col": [1, 2, 3]}
        self.dpyr = DataFrame(base_df)
        self.polars = pl.DataFrame(base_df)

    def test_accessors(self):
        self.assertEqual(str(pl.col('a')), str(c.a))
        self.assertEqual(str(pl.col('a') > 1), str(c.a > 1))
        self.assertEqual(str(pl.col('test col')), str(c.test__col))

    def test_select(self):
        dpyr_result = self.dpyr | select(c.a)
        polars_result = self.polars.select("a")
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_filter(self):
        dpyr_result = self.dpyr | filter(c.a > 1)
        polars_result = self.polars.filter(pl.col("a") > 1)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_mutate(self):
        dpyr_result = self.dpyr | mutate(c = c.a + c.b)
        polars_result = self.polars.with_columns(c=  pl.col("a") + pl.col("b"))
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_filter_and_mutate(self):
        dpyr_result = self.dpyr | filter(c.a > 1) \
            | mutate(c = c.a + c.b)
        polars_result = self.polars.filter(pl.col("a") > 1).with_columns(c=  pl.col("a") + pl.col("b"))
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_arrange(self):
        dpyr_result = self.dpyr | arrange(c.b)
        polars_result = self.polars.sort( "b")
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_arrange_desc(self):
        dpyr_result = self.dpyr | arrange(desc(c.b))
        polars_result = self.polars.sort("b", descending=True)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_head(self):
        dpyr_result = self.dpyr | head(2)
        polars_result = self.polars.head(2)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_tail(self):
        dpyr_result = self.dpyr | tail(2)
        polars_result = self.polars.tail(2)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_read_csv(self):
        dpyr_result = read_csv("iris.csv")
        polars_result = pl.read_csv("iris.csv")
        compare_dpyr_polars(dpyr_result, polars_result)


class TestDpyrVerbs(unittest.TestCase):
    def setUp(self):
        base_df = {
            "g": ["x", "y", "x", "y", "x"],
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
        self.dpyr = DataFrame(base_df)
        self.polars = pl.DataFrame(base_df)

    def test_group_by_returns_grouped(self):
        grouped = self.dpyr | group_by(c.g)
        self.assertIsInstance(grouped, GroupedDataFrame)
        self.assertEqual(grouped.keys, ["g"])

    def test_group_by_summarize(self):
        dpyr_result = self.dpyr | group_by(c.g) | summarize(total=c.a.sum())
        polars_result = self.polars.group_by("g", maintain_order=True).agg(total=pl.col("a").sum())
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_summarise_alias(self):
        dpyr_result = self.dpyr | group_by(c.g) | summarise(total=c.a.sum())
        polars_result = self.polars.group_by("g", maintain_order=True).agg(total=pl.col("a").sum())
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_summarize_with_n(self):
        dpyr_result = self.dpyr | group_by(c.g) | summarize(rows=n())
        polars_result = self.polars.group_by("g", maintain_order=True).agg(rows=pl.len())
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_summarize_ungrouped(self):
        dpyr_result = self.dpyr | summarize(total=c.a.sum())
        polars_result = self.polars.select(total=pl.col("a").sum())
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_grouped_mutate(self):
        dpyr_result = self.dpyr | group_by(c.g) | mutate(group_total=c.a.sum())
        polars_result = self.polars.with_columns(group_total=pl.col("a").sum().over("g"))
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_grouped_filter(self):
        dpyr_result = self.dpyr | group_by(c.g) | filter(c.a == c.a.max())
        polars_result = self.polars.filter((pl.col("a") == pl.col("a").max()).over("g"))
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_ungroup(self):
        result = self.dpyr | group_by(c.g) | ungroup()
        self.assertIsInstance(result, DataFrame)
        compare_dpyr_polars(result, self.polars)

    def test_distinct_all(self):
        df = DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        plf = pl.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        dpyr_result = df | distinct()
        polars_result = plf.unique(maintain_order=True)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_distinct_subset(self):
        df = DataFrame({"a": [1, 1, 2], "b": [3, 9, 4]})
        plf = pl.DataFrame({"a": [1, 1, 2], "b": [3, 9, 4]})
        dpyr_result = df | distinct(c.a)
        polars_result = plf.unique(subset=["a"], maintain_order=True).select(["a"])
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_distinct_keep_all(self):
        df = DataFrame({"a": [1, 1, 2], "b": [3, 9, 4]})
        plf = pl.DataFrame({"a": [1, 1, 2], "b": [3, 9, 4]})
        dpyr_result = df | distinct(c.a, keep_all=True)
        polars_result = plf.unique(subset=["a"], maintain_order=True)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_rename(self):
        dpyr_result = self.dpyr | rename(group=c.g)
        polars_result = self.polars.rename({"g": "group"})
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_count(self):
        dpyr_result = self.dpyr | count(c.g)
        polars_result = self.polars.group_by("g", maintain_order=True).agg(pl.len().alias("n"))
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_count_no_args(self):
        dpyr_result = self.dpyr | count()
        polars_result = self.polars.select(pl.len().alias("n"))
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_slice_sample_n(self):
        dpyr_result = self.dpyr | slice_sample(n=3, seed=1)
        self.assertEqual(dpyr_result.height, 3)

    def test_sample_n_alias(self):
        dpyr_result = self.dpyr | sample_n(n=2, seed=1)
        self.assertEqual(dpyr_result.height, 2)

    def test_pull(self):
        values = self.dpyr | pull(c.a)
        self.assertIsInstance(values, pl.Series)
        self.assertEqual(values.to_list(), [1, 2, 3, 4, 5])

    def test_lag_lead(self):
        dpyr_result = self.dpyr | mutate(prev=lag(c.a), nxt=lead(c.a))
        polars_result = self.polars.with_columns(
            prev=pl.col("a").shift(1),
            nxt=pl.col("a").shift(-1),
        )
        compare_dpyr_polars(dpyr_result, polars_result)


class TestDpyrJoins(unittest.TestCase):
    def setUp(self):
        left = {"id": [1, 2, 3], "x": ["a", "b", "c"]}
        right = {"id": [2, 3, 4], "y": ["p", "q", "r"]}
        self.dpyr_left = DataFrame(left)
        self.dpyr_right = DataFrame(right)
        self.pl_left = pl.DataFrame(left)
        self.pl_right = pl.DataFrame(right)

    def test_inner_join(self):
        dpyr_result = self.dpyr_left | inner_join(self.dpyr_right, by=c.id)
        polars_result = self.pl_left.join(self.pl_right, on="id", how="inner")
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_left_join(self):
        dpyr_result = self.dpyr_left | left_join(self.dpyr_right, by=c.id)
        polars_result = self.pl_left.join(self.pl_right, on="id", how="left")
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_right_join(self):
        dpyr_result = self.dpyr_left | right_join(self.dpyr_right, by=c.id)
        polars_result = self.pl_left.join(self.pl_right, on="id", how="right")
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_full_join(self):
        dpyr_result = self.dpyr_left | full_join(self.dpyr_right, by="id")
        polars_result = self.pl_left.join(self.pl_right, on="id", how="full", coalesce=True)
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_semi_join(self):
        dpyr_result = self.dpyr_left | semi_join(self.dpyr_right, by=c.id)
        polars_result = self.pl_left.join(self.pl_right, on="id", how="semi")
        compare_dpyr_polars(dpyr_result, polars_result)

    def test_anti_join(self):
        dpyr_result = self.dpyr_left | anti_join(self.dpyr_right, by=c.id)
        polars_result = self.pl_left.join(self.pl_right, on="id", how="anti")
        compare_dpyr_polars(dpyr_result, polars_result)


class TestDplyrFidelity(unittest.TestCase):
    """Behaviours where a naive polars port would diverge from dplyr."""

    def setUp(self):
        base_df = {
            "g": ["x", "x", "y", "y", "x"],
            "a": [3, 1, 2, 5, 1],
        }
        self.dpyr = DataFrame(base_df)
        self.polars = pl.DataFrame(base_df)

    def test_mutate_is_sequential(self):
        # dplyr evaluates mutate expressions in order; b is visible to d.
        result = self.dpyr | mutate(b=c.a + 1, d=c.b * 2)
        self.assertEqual(result.get_column("d").to_list(), [8, 4, 6, 12, 4])

    def test_arrange_desc_puts_nulls_last(self):
        df = DataFrame({"a": [3, None, 1]})
        result = df | arrange(desc(c.a))
        self.assertEqual(result.get_column("a").to_list(), [3, 1, None])

    def test_grouped_mutate_keeps_grouping(self):
        result = self.dpyr | group_by(c.g) | mutate(b=c.a + 1)
        self.assertIsInstance(result, GroupedDataFrame)
        self.assertEqual(result.keys, ["g"])

    def test_grouped_filter_keeps_grouping(self):
        result = self.dpyr | group_by(c.g) | filter(c.a > 1)
        self.assertIsInstance(result, GroupedDataFrame)

    def test_summarize_single_group_is_ungrouped(self):
        result = self.dpyr | group_by(c.g) | summarize(total=c.a.sum())
        self.assertIsInstance(result, DataFrame)
        self.assertNotIsInstance(result, GroupedDataFrame)

    def test_summarize_peels_last_group_level(self):
        df = DataFrame(
            {
                "g1": ["a", "a", "b", "b"],
                "g2": ["x", "y", "x", "y"],
                "v": [1, 2, 3, 4],
            }
        )
        result = df | group_by(c.g1, c.g2) | summarize(total=c.v.sum())
        self.assertIsInstance(result, GroupedDataFrame)
        self.assertEqual(result.keys, ["g1"])

    def test_group_by_replaces_by_default(self):
        grouped = self.dpyr | group_by(c.g) | group_by(c.a)
        self.assertEqual(grouped.keys, ["a"])

    def test_group_by_add_appends(self):
        grouped = self.dpyr | group_by(c.g) | group_by(c.a, add=True)
        self.assertEqual(grouped.keys, ["g", "a"])

    def test_count_on_grouped_keeps_original_grouping(self):
        result = self.dpyr | group_by(c.g) | count(c.a)
        self.assertIsInstance(result, GroupedDataFrame)
        self.assertEqual(result.keys, ["g"])

    def test_grouped_row_number_restarts_per_group(self):
        result = self.dpyr | group_by(c.g) | mutate(rn=row_number()) | arrange(c.g)
        # group x has 3 rows (1,2,3), group y has 2 rows (1,2)
        self.assertEqual(sorted(result.get_column("rn").to_list()), [1, 1, 2, 2, 3])

    def test_min_rank(self):
        result = self.dpyr | mutate(r=min_rank(c.a))
        polars_result = self.polars.with_columns(
            r=pl.col("a").rank(method="min").cast(pl.Int64)
        )
        compare_dpyr_polars(result, polars_result)

    def test_dense_rank(self):
        result = self.dpyr | mutate(r=dense_rank(c.a))
        polars_result = self.polars.with_columns(
            r=pl.col("a").rank(method="dense").cast(pl.Int64)
        )
        compare_dpyr_polars(result, polars_result)

    def test_row_number_of_column(self):
        result = self.dpyr | mutate(r=row_number(c.a))
        polars_result = self.polars.with_columns(
            r=pl.col("a").rank(method="ordinal").cast(pl.Int64)
        )
        compare_dpyr_polars(result, polars_result)


class TestSanitizeColName(unittest.TestCase):
    def test_plain_identifier(self):
        self.assertEqual(_sanitize_col_name("sepal_length"), "sepal_length")

    def test_spaces_become_underscores(self):
        self.assertEqual(_sanitize_col_name("my col"), "my_col")

    def test_hyphens_become_underscores(self):
        self.assertEqual(_sanitize_col_name("a-b"), "a_b")

    def test_leading_digit_prefixed(self):
        self.assertEqual(_sanitize_col_name("2fast"), "_2fast")

    def test_special_chars_stripped(self):
        self.assertEqual(_sanitize_col_name("a!b@c"), "abc")

    def test_multiple_spaces_collapsed(self):
        self.assertEqual(_sanitize_col_name("a  b"), "a_b")

    def test_empty_after_strip_becomes_col(self):
        self.assertEqual(_sanitize_col_name("!!!"), "_col")


class TestColumnsNamespace(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame({
            "sepal_length": [5.1, 4.9, 4.7],
            "my col": [1.4, 1.4, 1.3],
            "2fast": [0.2, 0.2, 0.2],
        })
        self.cols = self.df.cols

    def test_cols_returns_columns_instance(self):
        self.assertIsInstance(self.df.cols, Columns)

    def test_attribute_plain_name(self):
        self.assertEqual(str(self.cols.sepal_length), str(pl.col("sepal_length")))

    def test_attribute_sanitized_space(self):
        self.assertEqual(str(self.cols.my_col), str(pl.col("my col")))

    def test_attribute_sanitized_leading_digit(self):
        self.assertEqual(str(self.cols._2fast), str(pl.col("2fast")))

    def test_bracket_original_name(self):
        self.assertEqual(str(self.cols["my col"]), str(pl.col("my col")))

    def test_attribute_raises_for_missing(self):
        with self.assertRaises(AttributeError):
            _ = self.cols.nonexistent

    def test_bracket_raises_for_missing(self):
        with self.assertRaises(KeyError):
            _ = self.cols["nonexistent"]

    def test_dir_lists_sanitized_names(self):
        available = dir(self.cols)
        self.assertIn("sepal_length", available)
        self.assertIn("my_col", available)
        self.assertIn("_2fast", available)

    def test_works_in_filter_pipeline(self):
        result = self.df | filter(self.cols.sepal_length > 5.0)
        self.assertEqual(result.height, 1)

    def test_bracket_works_in_pipeline(self):
        result = self.df | filter(self.cols["my col"] > 1.3)
        self.assertEqual(result.height, 2)


class TestReadCsvInject(unittest.TestCase):
    _INJECT_COL = "_dpyr_test_unique_injected_xyz"
    _COLLISION_COL = "_dpyr_test_collision_abc"
    _TMP_CSV = "/tmp/dpyr_inject_test.csv"
    _COLLISION_CSV = "/tmp/dpyr_collision_test.csv"

    def setUp(self):
        import os
        pl.DataFrame({self._INJECT_COL: [1, 2, 3]}).write_csv(self._TMP_CSV)
        pl.DataFrame({self._COLLISION_COL: [1, 2, 3]}).write_csv(self._COLLISION_CSV)
        # Remove the test column from globals if a prior test left it
        globals().pop(self._INJECT_COL, None)

    def tearDown(self):
        import os
        globals().pop(self._INJECT_COL, None)
        globals().pop(self._COLLISION_COL, None)
        for f in [self._TMP_CSV, self._COLLISION_CSV]:
            if os.path.exists(f):
                os.remove(f)

    def test_inject_true_adds_col_to_globals(self):
        self.assertNotIn(self._INJECT_COL, globals())
        read_csv(self._TMP_CSV, inject=True)
        self.assertIn(self._INJECT_COL, globals())
        self.assertEqual(str(globals()[self._INJECT_COL]), str(pl.col(self._INJECT_COL)))

    def test_inject_false_does_not_add_to_globals(self):
        self.assertNotIn(self._INJECT_COL, globals())
        read_csv(self._TMP_CSV, inject=False)
        self.assertNotIn(self._INJECT_COL, globals())

    def test_inject_skips_existing_global_with_warning(self):
        import warnings
        globals()[self._COLLISION_COL] = "sentinel"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_csv(self._COLLISION_CSV, inject=True)
        self.assertTrue(any("already exists" in str(warning.message) for warning in w))
        self.assertEqual(globals()[self._COLLISION_COL], "sentinel")

    def test_inject_skips_python_keywords(self):
        import warnings
        tmp_csv = "/tmp/dpyr_keyword_test.csv"
        pl.DataFrame({"for": [1, 2, 3]}).write_csv(tmp_csv)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = read_csv(tmp_csv, inject=True)
            self.assertTrue(any("keyword" in str(warning.message).lower() for warning in w))
            # keyword column still accessible via df.cols
            self.assertEqual(str(df.cols["for"]), str(pl.col("for")))
        finally:
            import os
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    def test_inject_returns_dataframe(self):
        df = read_csv(self._TMP_CSV, inject=True)
        self.assertIsInstance(df, DataFrame)
