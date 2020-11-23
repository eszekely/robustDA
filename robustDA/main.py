#!/usr/bin/env python3

"""

Parameters
--------
Variables: tas, pr
Temporal resolutions: ann, mon, day
Grid size at 2.5 degrees
Scenarios CMIP6: historical, hist-GHG, hist-aer, hist-nat, piControl,
                 ssp119, ssp126, ssp245, ssp370, ssp434, ssp460, ssp585,
                 (1pctCO2, abrupt-2xCO2, abrupt-4xCO2,
                 esm-hist, esm-ssp585, esm-1pctCO2, hist-bgc, land-hist,
                 ssp534-over, ssp534-over-bgc, ssp585-bgc, 1pctCO2-bgc)
Date for CMIP6 starting at 1850
Target: 1) forcing (from ERF files): total, total_anthropogenic, GHG, CO2,
                                     aerosols, total_natural, volcanic, solar
        2) forced response (computed from the one forcing runs):
                                     hist-aer, hist-GHG, hist-nat
"""

import argparse

from anchor_regression import run_anchor_regression_all
from parse_args import args_climate, args_anchor


def main(params_climate, params_anchor):

    run_anchor_regression_all(params_climate, params_anchor, display_CV_plot = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--target", help="The target forcing you want to predict")
    parser.add_argument("--anchor", help="The anchor forcing we want to protect against")
    parser.add_argument("--gamma", nargs = "*", type = int, help="The anchor causal regularization parameter")
    parser.add_argument("--nonlinear_anchors", nargs = "*", help="The nonlinear functions used in anchor")

    args = parser.parse_args()
    params_climate = args_climate(args)
    params_anchor = args_anchor(args)
    
    main(params_climate, params_anchor)
