import numpy as np
from sklearn.model_selection import train_test_split


class ModelDF(object):

    def __init__(self, df, dep_var, id_var, indep_var, ix = None):
        """

        :param df: examples for training/testing
        :param dep_var:  dep variable name
        :param id_var: list of id variable names
        :param indep_var: list of independent variable names
        """
        self.df = df
        self.dep_var = dep_var
        self.id_var = id_var
        self.indep_var = indep_var


        self.y = df[dep_var]
        self.ID = df[id_var]
        self.X = df[indep_var]

        self.train = None
        self.test = None

    def _new_subset(self, ix):
        """For training and testing sets, we need to update all values, preserving any changes to y,ID,X"""
        new_subset_df = ModelDF(self.df, self.dep_var, self.id_var, self.indep_var)
        new_subset_df.y = self.y.iloc[ix]
        new_subset_df.ID = self.ID.iloc[ix]
        new_subset_df.X = self.X.iloc[ix]
        return new_subset_df

    def train_test_split(self, **options):
        """

        :param options: sklearn.model_selection.train_test_split parameters
        :return:
        """

        train_ix, test_ix = train_test_split(np.arange(self.df.shape[0]), **options)
        self.train = self._new_subset(train_ix)
        self.test = self._new_subset(test_ix)
        # self.X_train
        # self.X_test = ModelDF(self, self.X_train, self.dep_var, self.id_var, self.indep_var)
        # self.y_train = ModelDF(self, self.X_train, self.dep_var, self.id_var, self.indep_var)
        # self.y_test = ModelDF(self, self.X_train, self.dep_var, self.id_var, self.indep_var)

