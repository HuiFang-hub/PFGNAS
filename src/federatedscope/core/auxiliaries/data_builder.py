import logging
from src.federatedscope.core.auxiliaries.splitter_builder import get_splitter
from importlib import import_module
from src.federatedscope.core.data.utils import RegexInverseMap, load_dataset, \
    convert_data_mode
from src.federatedscope.core.auxiliaries.utils import setup_seed
from src.FLAGNNS.utils import random_supermask
import src.federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from src.federatedscope.contrib.data import *
except ImportError as error:
    logger.warning(
        f'{error} in `src.federatedscope.contrib.data`, some modules are not '
        f'available.')

# TODO: Add PyGNodeDataTranslator and PyGLinkDataTranslator
# TODO: move splitter to PyGNodeDataTranslator and PyGLinkDataTranslator
TRANS_DATA_MAP = {
    'BaseDataTranslator': [
        '.*?@.*?', 'hiv', 'proteins', 'imdb-binary', 'bbbp', 'tox21', 'bace',
        'sider', 'clintox', 'esol', 'freesolv', 'lipo', 'cifar4cl', 'cifar4lp'
    ],
    'DummyDataTranslator': [
        'toy', 'quadratic', 'femnist', 'celeba', 'shakespeare', 'twitter',
        'subreddit', 'synthetic', 'ciao', 'epinions', '.*?vertical_fl_data.*?',
        '.*?movielens.*?', '.*?cikmcup.*?', 'graph_multi_domain.*?', 'cora',
        'citeseer', 'pubmed', 'dblp_conf', 'dblp_org', 'csbm.*?', 'fb15k-237',
        'wn18', 'adult', 'abalone', 'credit', 'blog','amazon','reddit','flickr','yelp','facebook','twitch',
        'tfinance','tsocial','lls'
    ],  # Dummy for FL dataset
    'RawDataTranslator': ['hetero_nlp_tasks'],
}
DATA_TRANS_MAP = RegexInverseMap(TRANS_DATA_MAP, None)
NUM_POP = 60

def get_data(config, client_cfgs=None):
    """Instantiate the data and update the configuration accordingly if
    necessary.

    Arguments:
        config: a cfg node object
        client_cfgs: dict of client-specific cfg node object
    Returns:
        The dataset object and the updated configuration.

    Note:
      The available ``data.type`` is shown below:
        ==================================  ===========================
        Data type                           Domain
        ==================================  ===========================
        FEMNIST	                            CV
        Celeba	                            CV
        ``${DNAME}@torchvision``	        CV
        Shakespeare	                        NLP
        SubReddit	                        NLP
        Twitter (Sentiment140)	            NLP
        ``${DNAME}@torchtext``	            NLP
        ``${DNAME}@huggingface_datasets``  	NLP
        Cora	                            Graph (node-level)
        CiteSeer	                        Graph (node-level)
        PubMed	                            Graph (node-level)
        DBLP_conf	                        Graph (node-level)
        DBLP_org	                        Graph (node-level)
        csbm	                            Graph (node-level)
        Reddit                              Graph (node-level)
        Karate Club                         Graph (node-level)
        Protein-Protein Interaction (PPI)   Graph (node-level)
        Epinions	                        Graph (link-level)
        Ciao	                            Graph (link-level)
        FB15k	                            Graph (link-level)
        FB15k-237	                        Graph (link-level)
        WN18	                            Graph (link-level)
        MUTAG	                            Graph (graph-level)
        BZR	                                Graph (graph-level)
        COX2	                            Graph (graph-level)
        DHFR	                            Graph (graph-level)
        PTC_MR	                            Graph (graph-level)
        AIDS	                            Graph (graph-level)
        NCI1	                            Graph (graph-level)
        ENZYMES	                            Graph (graph-level)
        DD	                                Graph (graph-level)
        PROTEINS	                        Graph (graph-level)
        COLLAB	                            Graph (graph-level)
        IMDB-BINARY	                        Graph (graph-level)
        IMDB-MULTI	                        Graph (graph-level)
        REDDIT-BINARY	                    Graph (graph-level)
        HIV	                                Graph (graph-level)
        ESOL	                            Graph (graph-level)
        FREESOLV	                        Graph (graph-level)
        LIPO	                            Graph (graph-level)
        PCBA	                            Graph (graph-level)
        MUV	                                Graph (graph-level)
        BACE	                            Graph (graph-level)
        BBBP	                            Graph (graph-level)
        TOX21	                            Graph (graph-level)
        TOXCAST	                            Graph (graph-level)
        SIDER	                            Graph (graph-level)
        CLINTOX	                            Graph (graph-level)
        graph_multi_domain_mol	            Graph (graph-level)
        graph_multi_domain_small	        Graph (graph-level)
        graph_multi_domain_biochem	        Graph (graph-level)
        cikmcup	                            Graph (graph-level)
        toy	                                Tabular
        synthetic	                        Tabular
        quadratic	                        Tabular
        ``${DNAME}openml``	                Tabular
        vertical_fl_data	                Tabular(vertical)
        VFLMovieLens1M	                    Recommendation
        VFLMovieLens10M	                    Recommendation
        HFLMovieLens1M	                    Recommendation
        HFLMovieLens10M	                    Recommendation
        VFLNetflix	                        Recommendation
        HFLNetflix	                        Recommendation
        ==================================  ===========================
    """
    # Fix the seed for data generation
    setup_seed(12345)

    # for func in register.data_dict.values():
    #     data_and_config = func(config, client_cfgs)
    #     if data_and_config is not None:
    #         return data_and_config
    # dataset, modified_config = load_dataset(config, client_cfgs)

    if config.data.type.lower() in register.data_dict.keys():
        func = register.data_dict[config.data.type.lower()]
        data_and_config = func(config, client_cfgs)
        dataset, modified_config = data_and_config
        # splitter = get_splitter(config)
        # dataset = splitter(dataset[0])
    else:
        dataset, modified_config = load_dataset(config, client_cfgs) #{dict:6}
    # Load dataset from source files


    # Apply translator to non-FL dataset to transform it into its federated
    # counterpart
    test =DATA_TRANS_MAP[config.data.type.lower()]
    translator = getattr(import_module('src.federatedscope.core.data'),
                         DATA_TRANS_MAP[config.data.type.lower()])(
                             modified_config, client_cfgs)
    dataset = translator(dataset)

    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    data = convert_data_mode(dataset, modified_config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)
    supermasks = [random_supermask() for i in range(NUM_POP)]
    modified_config.merge_from_list(['data.supermasks', supermasks])
    return data, modified_config

