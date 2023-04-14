def get_envs():
    from torch.utils import collect_env

    return collect_env.get_pretty_env_info()
