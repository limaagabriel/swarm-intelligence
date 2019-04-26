class StopCriterionDefinition(object):
    def __call__(self, **kwargs):
        return kwargs[self.keyword] > self.limit

    def count(self, **kwargs):
        return kwargs[self.keyword]


class StopCriterion(object):
    @staticmethod
    def iteration_limit(max_iterations):
        class IterativeStopCriterion(StopCriterionDefinition):
            limit = max_iterations
            keyword = 'iterations'

        return IterativeStopCriterion()

    @staticmethod
    def fn_evaluation(max_evaluations):
        class FnEvaluationStopCriterion(StopCriterionDefinition):
            keyword = 'evaluations'
            limit = max_evaluations

        return FnEvaluationStopCriterion()
