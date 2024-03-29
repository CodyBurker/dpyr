import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
from dpyr import DataFrame, filter, select, mutate, c, arrange, head, read_csv
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
    
    def test_head(self):
        dpyr_result = self.dpyr | head(2)
        polars_result = self.polars.head(2)
        compare_dpyr_polars(dpyr_result, polars_result)
    
    def test_read_csv(self):
        dpyr_result = read_csv("iris.csv")
        polars_result = pl.read_csv("iris.csv")
        compare_dpyr_polars(dpyr_result, polars_result)
        