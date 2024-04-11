from src.federatedscope.core.fed_runner import StandaloneRunner, DistributedRunner,pf_StandaloneRunner


def get_runner(data, server_class, client_class, config, client_configs=None):
    """
    Instantiate a runner based on a configuration file

    Args:
        data: ``core.data.StandaloneDataDict`` in standalone mode, \
            ``core.data.ClientData`` in distribute mode
        server_class: server class
        client_class: client class
        config: configurations for FL, see ``src.federatedscope.core.configs``
        client_configs: client-specific configurations

    Returns:
        An instantiated FedRunner to run the FL course.

    Note:
      The key-value pairs of built-in runner and source are shown below:
        ===============================  ==============================
        Mode                             Source
        ===============================  ==============================
        ``standalone``                   ``core.fed_runner.StandaloneRunner``
        ``distributed``                  ``core.fed_runner.DistributedRunner``
        ===============================  ==============================
    """

    mode = config.federate.mode.lower()

    if mode == 'standalone':
        if config.model.type in ['pfgnas', 'pfgnas-random','pfgnas-evo']:
            runner_cls = pf_StandaloneRunner
        else:

            runner_cls = StandaloneRunner
    elif mode == 'distributed':
        runner_cls = DistributedRunner

    return runner_cls(data=data,
                      server_class=server_class,
                      client_class=client_class,
                      config=config,
                      client_configs=client_configs)
