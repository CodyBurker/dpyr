"""End-to-end integration tests for dpyr.

These exercise realistic multi-verb pipelines on the iris dataset and confirm
that a dpyr pipeline produces the same result as the equivalent hand-written
polars pipeline. Where the dpyr pipeline mirrors a dplyr idiom, the polars
comparison is written to match dplyr's semantics (e.g. coalesced full-join keys,
grouped summaries in group order).
"""
import unittest

import polars as pl
from pandas._testing import assert_frame_equal

from dpyr import (
    DataFrame,
    read_csv,
    select,
    filter,
    mutate,
    arrange,
    desc,
    head,
    tail,
    group_by,
    ungroup,
    summarize,
    distinct,
    rename,
    count,
    pull,
    left_join,
    inner_join,
    full_join,
    anti_join,
    n,
    lag,
    lead,
    c,
)


def assert_equal(dpyr_result, polars_result):
    """Compare a dpyr frame to a polars frame, ignoring column-name cosmetics."""
    pandas_dpyr = dpyr_result.to_pandas()
    pandas_polars = polars_result.to_pandas()
    pandas_polars.columns = pandas_dpyr.columns
    assert_frame_equal(pandas_dpyr, pandas_polars)


class TestIrisPipelines(unittest.TestCase):
    """Full pipelines over the iris dataset (column dots become underscores)."""

    def setUp(self):
        # Raw polars frame keeps the original dotted names; dpyr normalises them.
        self.pl_iris = pl.read_csv("iris.csv")

    def test_filter_select_mutate_arrange_head(self):
        # The canonical demo pipeline: Setosa rows, sepal ratio, top 5.
        result = (
            read_csv("iris.csv")
            | filter(c.variety == "Setosa")
            | select(c.sepal_length, c.sepal_width, c.variety)
            | mutate(sepal_ratio=c.sepal_length / c.sepal_width)
            | arrange(desc(c.sepal_ratio))
            | head(5)
        )
        expected = (
            self.pl_iris.filter(pl.col("variety") == "Setosa")
            .select("sepal.length", "sepal.width", "variety")
            .with_columns(
                sepal_ratio=pl.col("sepal.length") / pl.col("sepal.width")
            )
            .sort("sepal_ratio", descending=True)
            .head(5)
        )
        assert_equal(result, expected)

    def test_group_by_summarize_arrange(self):
        # Mean petal length per variety, sorted descending.
        result = (
            read_csv("iris.csv")
            | group_by(c.variety)
            | summarize(mean_petal=c.petal_length.mean(), rows=n())
            | arrange(desc(c.mean_petal))
        )
        expected = (
            self.pl_iris.group_by("variety", maintain_order=True)
            .agg(
                mean_petal=pl.col("petal.length").mean(),
                rows=pl.len(),
            )
            .sort("mean_petal", descending=True)
        )
        assert_equal(result, expected)

    def test_grouped_mutate_then_filter(self):
        # Flag rows above their variety's mean sepal length, keep just those.
        result = (
            read_csv("iris.csv")
            | group_by(c.variety)
            | mutate(variety_mean=c.sepal_length.mean())
            | filter(c.sepal_length > c.variety_mean)
            | select(c.variety, c.sepal_length, c.variety_mean)
        )
        expected = (
            self.pl_iris.with_columns(
                variety_mean=pl.col("sepal.length").mean().over("variety")
            )
            .filter(pl.col("sepal.length") > pl.col("variety_mean"))
            .select("variety", "sepal.length", "variety_mean")
        )
        assert_equal(result, expected)

    def test_count_and_arrange(self):
        result = (
            read_csv("iris.csv")
            | count(c.variety)
            | arrange(desc(c.n))
        )
        expected = (
            self.pl_iris.group_by("variety", maintain_order=True)
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        assert_equal(result, expected)

    def test_distinct_variety(self):
        result = read_csv("iris.csv") | distinct(c.variety)
        expected = self.pl_iris.select("variety").unique(maintain_order=True)
        assert_equal(result, expected)

    def test_rename_and_pull(self):
        values = (
            read_csv("iris.csv")
            | rename(species=c.variety)
            | distinct(c.species)
            | pull(c.species)
        )
        self.assertEqual(values.to_list(), ["Setosa", "Versicolor", "Virginica"])

    def test_lag_lead_window(self):
        result = (
            read_csv("iris.csv")
            | head(5)
            | select(c.sepal_length)
            | mutate(prev=lag(c.sepal_length), nxt=lead(c.sepal_length))
        )
        expected = (
            self.pl_iris.head(5)
            .select("sepal.length")
            .with_columns(
                prev=pl.col("sepal.length").shift(1),
                nxt=pl.col("sepal.length").shift(-1),
            )
        )
        assert_equal(result, expected)

    def test_tail(self):
        result = read_csv("iris.csv") | tail(3) | select(c.variety)
        expected = self.pl_iris.tail(3).select("variety")
        assert_equal(result, expected)


class TestJoinPipelines(unittest.TestCase):
    """Multi-step pipelines combining joins with other verbs."""

    def setUp(self):
        self.orders = DataFrame(
            {
                "customer_id": [1, 2, 1, 3, 2],
                "amount": [100, 200, 50, 300, 75],
            }
        )
        self.customers = DataFrame(
            {
                "customer_id": [1, 2, 4],
                "name": ["Alice", "Bob", "Dana"],
            }
        )
        self.pl_orders = pl.DataFrame(
            {
                "customer_id": [1, 2, 1, 3, 2],
                "amount": [100, 200, 50, 300, 75],
            }
        )
        self.pl_customers = pl.DataFrame(
            {
                "customer_id": [1, 2, 4],
                "name": ["Alice", "Bob", "Dana"],
            }
        )

    def test_join_then_group_summarize(self):
        # Total spend per named customer.
        result = (
            self.orders
            | inner_join(self.customers, by=c.customer_id)
            | group_by(c.name)
            | summarize(total=c.amount.sum())
            | arrange(c.name)
        )
        expected = (
            self.pl_orders.join(self.pl_customers, on="customer_id", how="inner")
            .group_by("name", maintain_order=True)
            .agg(total=pl.col("amount").sum())
            .sort("name")
        )
        assert_equal(result, expected)

    def test_left_join_keeps_all_orders(self):
        result = self.orders | left_join(self.customers, by=c.customer_id)
        expected = self.pl_orders.join(
            self.pl_customers, on="customer_id", how="left"
        )
        assert_equal(result, expected)
        self.assertEqual(result.height, self.pl_orders.height)

    def test_anti_join_finds_orphans(self):
        # Orders whose customer is not in the customers table (customer_id 3).
        result = (
            self.orders
            | anti_join(self.customers, by=c.customer_id)
            | distinct(c.customer_id)
        )
        self.assertEqual(result.get_column("customer_id").to_list(), [3])

    def test_full_join_coalesces_keys(self):
        # dplyr-style full join: a single coalesced key column.
        result = self.orders | full_join(self.customers, by=c.customer_id)
        self.assertIn("customer_id", result.columns)
        self.assertNotIn("customer_id_right", result.columns)
        # customer 4 (no orders) and customer 3 (no profile) both appear.
        ids = set(result.get_column("customer_id").to_list())
        self.assertTrue({1, 2, 3, 4}.issubset(ids))


if __name__ == "__main__":
    unittest.main()
