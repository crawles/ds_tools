import pandas as pd
import pyspark.sql.functions as F

from pyspark.sql.types import *

def spark_udf(spark_type):
    """
    Decorator for creating pyspark functions:

    Example
    -------
    @spark_udf(return_type)
    def col_plus_one(num_col):
        return num_col+1

    sdf.select(col_plus_one('col_a'))
    """
    def function_decorator(func):
        return F.UserDefinedFunction(func, spark_type)
    return function_decorator


def _apply_to_struct(fxn, return_type, struct_col, **fxn_kwargs):
    """
    :param group_col: column name of a pyspark StructType
    :param fxn: function to apply to a Pandas DataFrame
    :param return_type: pyspark return type
    :return:
    """
    @spark_udf(return_type)
    def _apply_fxn(_struct_col):
        first_row = _struct_col[0]
        col_names = first_row.__fields__
        df = pd.DataFrame(_struct_col, columns=col_names)
        return fxn(df,**fxn_kwargs)

    return _apply_fxn(struct_col)


def apply_py_fxn(fxn, return_type, *cols, **fxn_kwargs):
    print(fxn_kwargs)
    struct_args = [F.col(c) for c in cols]
    struct = F.struct(*struct_args)
    struct_list = F.collect_list(struct)
    return _apply_to_struct(fxn, return_type, struct_list, **fxn_kwargs).alias(fxn.__name__)










































##
