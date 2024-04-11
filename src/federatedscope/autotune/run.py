def run_scheduler(scheduler, cfg, client_cfgs=None):
    """
    This function is to optimize FedHPO problem by scheduler. The method is
    decided by  `cfg.hpo.scheduler`.
    Args:
        scheduler: Scheduler for conducting serval FS runs, \
            see ``src.federatedscope.autotune.algos.Scheduler``
        cfg: The configurations of the FL course.
        client_cfgs: The clients' configurations.
    """
    if cfg.hpo.scheduler in ['sha', 'wrap_sha']:
        _ = scheduler.optimize()
    elif cfg.hpo.scheduler in [
            'rs', 'bo_kde', 'hb', 'bohb', 'wrap_rs', 'wrap_bo_kde', 'wrap_hb',
            'wrap_bohb'
    ]:
        from src.federatedscope.autotune.hpbandster import run_hpbandster
        run_hpbandster(cfg, scheduler, client_cfgs)
    elif cfg.hpo.scheduler in ['bo_gp', 'bo_rf', 'wrap_bo_gp', 'wrap_bo_rf']:
        from src.federatedscope.autotune.smac import run_smac
        run_smac(cfg, scheduler, client_cfgs)
    else:
        raise ValueError(f'No scheduler named {cfg.hpo.scheduler}')
