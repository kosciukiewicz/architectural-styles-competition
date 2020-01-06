import itertools
import pandas as pd
import os


class ExperimentRunner(object):
    def __init__(self, name, params, metrics, output_path):
        self.name = name
        self.params = {param.name: param for param in params}
        self.metrics = metrics
        self.runs = []
        self.output_path = output_path
        self.results_callbacks = []

    def run_experiment(self, experiment_runner):
        self.runs.append(experiment_runner)

    def run(self):
        for run in self.runs:
            runner_result = run.run(self.metrics, self.params,
                                    {param.name: param.default_value for param
                                     in self.params.values()})
            run.run_results_callbacks(self.results_callbacks, runner_result)
            self.save_runner_results(run.name, runner_result)

    def save_runner_results(self, runner_name, runner_results):
        self.check_if_path_exists_and_create(f"{self.output_path}/{self.name}")
        to_ignore = [param.name for param in self.params.values() if
                     param.ignore_in_reults]
        to_save = [column for column in runner_results.columns if
                   column not in to_ignore]

        runner_results = runner_results[to_save]
        runner_results.to_csv(
            f"{self.output_path}/{self.name}/{runner_name}.csv")

    def check_if_path_exists_and_create(self, path_to_check):
        if not os.path.exists(path_to_check):
            os.makedirs(path_to_check)

    def add_results_callback(self, callback):
        self.results_callbacks.append(callback)


class Runner(object):
    def __init__(self, name, run_function, experiment_params, default_params,
                 results_callbacks=[], metrics=None):
        self.name = name
        self.run_function = run_function
        self.experiment_params = experiment_params
        self.default_params = default_params
        self.metrics = metrics
        self.results_callbacks = results_callbacks

    def run(self, metrics, experiment_params, experiment_default_params):
        run_results = []
        params_values = self.generate_params_values(experiment_default_params)
        for params in params_values:
            params_values = self.get_experiment_params_values(params,
                                                              experiment_params)
            self.print_info(params_values)
            results = self.run_function(params_values)

            metrics_results = self.generate_metrics(results,
                                                    self.metrics if self.metrics else metrics)

            self.run_results_callbacks(self.results_callbacks, results)
            run_result = params
            run_result.update(metrics_results)
            run_results.append(run_result)

        return pd.DataFrame(run_results)

    def get_experiment_params_values(self, run_params, experiment_params):
        new_params = {}
        for param_name, param_value in run_params.items():
            if experiment_params[param_name].has_custom_values():
                new_params[param_name] = \
                    experiment_params[param_name].values_dict[param_value]
            else:
                new_params[param_name] = param_value

        return new_params

    def generate_metrics(self, results, metrics):
        metric_results = {}
        for metric in metrics:
            metric_results[metric.name] = metric.run(results)
        return metric_results

    def generate_params_values(self, experiment_default_params):
        params_names = [k for k, v in self.experiment_params.items()]
        params_values = tuple([v for k, v in self.experiment_params.items()])
        params_values = list(itertools.product(*params_values))
        params_values = [
            {params_names[i]: param for i, param in enumerate(params)} for
            params in params_values]
        default_params = self.get_default_params(experiment_default_params)

        for params_value in params_values:
            params_value.update(default_params)

        return params_values

    def get_default_params(self, experiment_default_params):
        new_default_params = {}
        default_params = [(name, value) for name, value in
                          experiment_default_params.items()
                          if name not in self.experiment_params.keys()]

        for name, value in default_params:
            if name in self.default_params.keys():
                new_default_params[name] = self.default_params[name]
            else:
                new_default_params[name] = experiment_default_params[name]

        return new_default_params

    def run_results_callbacks(self, callbacks, results):
        for callback in callbacks:
            callback(self.name, results)

    def print_info(self, params):
        print("--------------------")
        print("New run with params: ")
        for name, value in params.items():
            print(f"{name}: {value}")


class MetricRunner(object):
    def __init__(self, name, run):
        self.name = name
        self.run = run


class Param(object):
    def __init__(self, name, default_value, values_dict=None, param_type=None,
                 ignore_in_results=False):
        self.name = name
        self.default_value = default_value
        self.param_type = None
        self.values_dict = values_dict
        self.ignore_in_reults = ignore_in_results

        if param_type:
            self.param_type = param_type
        else:
            self.param_type = type(default_value)

    def has_custom_values(self):
        return self.values_dict is not None
