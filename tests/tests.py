import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
from dpyr import *
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
        tests = [
            (pl.col('a'), c.a),
            (pl.col('a') > 1, c.a > 1),
            (pl.col('test col'), c.test__col)
        ]
        for i, (polars_expr, dpyr_expr) in enumerate(tests):
            with self.subTest(i=i):
                self.assertEqual(str(polars_expr), str(dpyr_expr))

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
    
    def test_head(self):
        dpyr_result = self.dpyr | head(2)
        polars_result = self.polars.head(2)
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_read_csv(self):
        dpyr_result = read_csv("iris.csv")
        polars_result = pl.read_csv("iris.csv")
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_distinct(self):
        dpyr_result = self.dpyr | distinct()  | arrange(c.a, c.b, c.test__col)
        polars_result = self.polars.unique().sort("a", "b", "test col")
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_distinct_subset(self):
        dpyr_result = (self.dpyr| select(c.a) | distinct(c.a) | arrange(c.a))
        polars_result = self.polars.select('a').unique(["a"]).sort("a")
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_distinct_errors(self):
        with self.assertRaises(ValueError):
            self.dpyr | distinct(c.a +  c.b)
    
    def test_rename(self):
        dpyr_result = self.dpyr | rename(new_name = c.a, new_name_2 = c.b, new_name_3 = "test col")
        polars_result = self.polars.rename({"a": "new_name", "b": "new_name_2", "test col": "new_name_3"})
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_rename_errors(self):
        with self.assertRaises(ValueError) as cm:
            self.dpyr | rename(new_name = c.nonexistent_col)
        self.assertEqual(str(cm.exception), "Column nonexistent_col does not exist")
    
    def test_count(self):
        dpyr_result = (self.dpyr | count())
        polars_result = self.polars.select(pl.len().alias('n'))
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_count_existing_col(self):
        dpyr_result = self.dpyr | count(c.a) 
        dpyr_result = dpyr_result.sort("a") # Don't want to use dpyr.sort in case it is broken
        polars_result = self.polars.group_by("a").len().sort("a")
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_count_new_col(self):
        dplyr_result = self.dpyr | count(new_col = c.a + c.b)
        polars_result = self.polars.group_by(pl.col("a") + pl.col("b")).len()
        compare_dpyr_polars(dplyr_result, polars_result)