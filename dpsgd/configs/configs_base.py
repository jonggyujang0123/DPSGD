from ml_collections.config_dict import ConfigDict

def BaseConfig():
    cfg = ConfigDict()
    cfg.lr = 1e-3
    cfg.agent = 'DPSGD'
    cfg.agent2 = ConfigDict()
    cfg.agent2.a =1
    cfg.agent2.b= 3
    return cfg
