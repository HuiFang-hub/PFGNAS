from src.federatedscope.core.configs.config import CN
from src.federatedscope.register import register_config



def llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Dataset related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN()
    
    cfg.llm.type = 'gpt4'
    cfg.llm.ablation = 1
        # llm model
    


    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_llm_cfg)


def assert_llm_cfg(cfg):
    pass
    # --------------------------------------------------------------------


register_config("llm_cfg", llm_cfg)
