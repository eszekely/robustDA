def args_climate(args):

    """ Default climate parameters """

    dict_param_climate = {
        "temporalRes": "ann",
        "variables": ["tas"],
        "scenarios": ["historical", "piControl"],
        "startDate": 1850,
        "endDate": 2014,
        "target": args.target,
        "anchor": args.anchor,
    }

    return dict_param_climate


def args_anchor(args):

    """ Default anchor regression parameters """

    dict_param_anchor = {
        "gamma": args.gamma,
        "h_anchors": args.nonlinear_anchors,
    }

    return dict_param_anchor
