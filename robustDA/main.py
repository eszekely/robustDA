#!/usr/bin/env python3

import argparse
import yaml

from anchor_regression import (
    run_anchor_regression_all,
    param_optimization,
    param_optimization_gamma,
    subagging,
)
from hypothesis_testing import test_DA


def args_params(args):

    """ Climate parameters """

    dict_param_climate = {
        "temporalRes": args.temporalRes,
        "variables": args.variables,
        "scenarios": args.scenarios,
        "startDate": args.startDate,
        "endDate": args.endDate,
        "target": args.target,
        "anchor": args.anchor,
    }

    """ Anchor parameters """

    dict_param_anchor = {
        "gamma": args.gamma,
        "h_anchors": args.nonlinear_anchors,
    }

    return dict_param_climate, dict_param_anchor


def parser_args(input_params):

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", help="The experiment to run")

    for arg, val in input_params.items():
        parser.add_argument(
            "-%s" % arg,
            type=type(val),
            default=val,
        )

    args = parser.parse_args()
    exp = args.exp
    params_climate, params_anchor = args_params(args)
    print(params_climate)
    print(params_anchor)

    return exp, params_climate, params_anchor


def get_parameters():

    with open("./../params.yml") as file:
        input_params = yaml.full_load(file)

    exp, params_climate, params_anchor = parser_args(input_params)

    return exp, params_climate, params_anchor


def main(exp, params_climate, params_anchor):

    if exp == "run_anchor_regression_all":
        run_anchor_regression_all(
            params_climate, params_anchor, display_CV_plot=True
        )
    elif exp == "param_opt":
        param_optimization(params_climate, params_anchor)
    elif exp == "param_opt_gamma":
        param_optimization_gamma(params_climate, params_anchor)
    elif exp == "subagging":
        nbRuns = 10
        subagging(params_climate, params_anchor, nbRuns)
    elif exp == "HT":
        test_DA(params_climate, params_anchor)


if __name__ == "__main__":

    exp, params_climate, params_anchor = get_parameters()
    main(exp, params_climate, params_anchor)
