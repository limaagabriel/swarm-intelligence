class Step(object):
    @staticmethod
    def linear(individual=(0.1, 0.001), volitive=(0.01, 0.001)):
        class LinearStep(object):
            def __init__(self):
                self.__step = {
                    'individual': individual[0],
                    'volitive': volitive[0]
                }

                self.__boundaries = {
                    'individual': individual,
                    'volitive': volitive
                }

            def __call__(self, step_id):
                return self.__step[step_id]

            def update(self, stop_criterion, **kwargs):
                for step_id in self.__boundaries:
                    boundaries = self.__boundaries[step_id]

                    a = boundaries[0]
                    b = (boundaries[0] - boundaries[1]) * stop_criterion.count(**kwargs)
                    c = stop_criterion.limit
                    self.__step[step_id] = a - (b / c)

        return LinearStep()
