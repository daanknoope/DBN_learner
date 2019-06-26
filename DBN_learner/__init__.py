from typing import List

name = "DBN_learner"

from gobnilp import *
from pandas import DataFrame


class DBN_Learner:
    def __init__(self, unroll, max_parents, ESS):
        self.unroll = unroll
        self.max_parents = max_parents
        self.ESS = ESS

    @staticmethod
    def create_arities(dds: DataFrame) -> str:
        lookback = create_lookback_dataset(dds, 1)
        return " ".join([str(int(max(lookback[x]) + 1)) for x in lookback.columns])

    @staticmethod
    def get_state_names(dss: DataFrame, unroll: int):
        unrolled = create_lookback_dataset(dss, unroll)
        return dict(zip(unrolled.columns,
                        map(lambda x: list(range(int(min(unrolled[x])), int(max(unrolled[x])) + 1)), unrolled.columns)))

    def learn_DBN(self, dataset: DataFrame, objective: str, time_vars: List[str]) -> BayesianModel:
        arities = self.create_arities(dataset)
        state_names = list(self.get_state_names(dataset, self.unroll).keys())

        additional_constraints = [(objective, objective + '_1', False)]  # this aint nice
        additional_constraints += [(time_var, x, False) for time_var, x in product(time_vars, state_names)]

        G = create_unrolled_dbn(dataset, self.unroll, ESS=self.ESS, arities=arities, parent_lim=self.max_parents,
                                additional_constraints=additional_constraints)
        G.fit(create_lookback_dataset(dataset, self.unroll))
        return G
