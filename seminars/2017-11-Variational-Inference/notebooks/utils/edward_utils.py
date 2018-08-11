

def get_edward_data(bayesian_params):
    posteriors = {}
    kl_scaling = {}
    for k, v in bayesian_params.items():
        for pk, pv in v['posteriors'].items():
            posteriors[pk] = pv
        for pk, pv in v['kl_scaling'].items():
            kl_scaling[pk] = pv

    return posteriors, kl_scaling