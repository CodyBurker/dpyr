import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
from dpyr import (
    DataFrame,
    GroupedDataFrame,
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
