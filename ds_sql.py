import re

import pandas as pd

class DateFeatures(object):

    def __init__(self, feature_types = {'day_of_week': 'u', 'month': 'M', 'day_of_month': 'd'}):
        self.feature_types = feature_types

    def create(self, cname):
        dt_sql = []
        for fname in self.feature_types:
            option = self.feature_types[fname]
            _params = {'cname':cname, 'option': option, 'fname': fname}
            fcalc = "date_format({cname}, '{option}') {cname}_{fname}".format(**_params)
            dt_sql.append(fcalc)
        return dt_sql

class DummyVariable():

    def __init__(self, default_null_value = 'NULL_VALUE'):
        self.default_null_value = default_null_value


    def _clean_dummy_val(self, cval):
        """For creating dummy variable columns, we need to remove special characters in values"""
        cval_clean = cval.replace(' ','_').replace('-','_').replace('"','').replace("'",'')
        return re.sub('[^a-zA-Z\d\s:]', '', cval_clean)

    def _sql_is_equal(self, col_name, col_value):
        return "COALESCE({col_name},'NULL_VALUE') = '{clean_col_val}'".format(col_name=col_name,
                                                                        clean_col_val=col_value)

    def _cast_to_int(self, col_name):
        return 'CAST({col_name} as INT)'.format(col_name=col_name)

    def _get_N_values(self, s, N):
        """Given a pandas series, we want the top values, then we want to make dummy variables."""
        s = s.fillna(self.default_null_value)
        ncols_vals = s.value_counts().shape[0]
        n_dummy_cols = max(1, min(ncols_vals-1, N))
        # fields = s.value_counts().index[0:n_dummy_cols].sort_values().tolist()
        value_counts_per = (s.value_counts() / s.value_counts().sum())[0:n_dummy_cols].sort_index()
        fields = value_counts_per.index.tolist()
        field_percentages = value_counts_per.values.tolist()
        return fields, field_percentages

    def _create_N_dummy_col(self, col_name, fields, field_percentages, N=10):
        """Given top N columns, we will make the sql code for getting the top values"""
        sql_dummy = []
        dummy_var_percentages = {}
        for v, per in zip(fields, field_percentages):
            col_equal = self._sql_is_equal(col_name, v)
            dummy_logic = self._cast_to_int(col_equal)
            _params = {'dummy_logic': dummy_logic,
                       'col_name': col_name,
                       'col_value': self._clean_dummy_val(v)}
            sql = '{dummy_logic} AS {col_name}_{col_value}'.format(**_params)
            dummy_var_percentages['{col_name}_{col_value}'.format(**_params)] = per
            sql_dummy.append(sql)
        return sql_dummy, dummy_var_percentages

    def map_dummy(self, s, N):
        col_name = s.name
        fields, field_percentages = self._get_N_values(s, N)
        return self._create_N_dummy_col(col_name, fields, field_percentages)


class AggFeatures(object):
    fxns = ['MIN({}) AS MIN_{}',
            'MAX({}) AS MAX_{}',
            'AVG({}) AS AVG_{}',
            'SUM({}) AS SUM_{}',
            'PERCENTILE(CAST({} AS BIGINT) AS MEDIAN_{}']

    def gen_features(self, feature, feature_name=None):
        if not feature_name:
            feature_name = feature
        return [self._apply_fxn(feature, fxn, feature_name) for fxn in self.fxns]

    def _apply_fxn(selfself, col, func, feature_name):
        return func.format(col, feature_name)

