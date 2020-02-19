from copy import deepcopy

from mlagents.envs import UnityEnvironment
from common.unity_wrapper import InfoWrapper, UnityReturnWrapper, SamplerWrapper, UnityReturnWrapper_GCN, InfoWrapper_GCN


def make_env(env_args, use_GCN):
    if env_args['type'] == 'gym':
        pass
    elif env_args['type'] == 'unity':
        env = make_unity_env(env_args, use_GCN)
    else:
        raise Exception('Unknown environment type.')
    return env





def make_unity_env(env_args, use_GCN):
    if env_args['file_path'] is None:
        env = UnityEnvironment()
    else:
        env = UnityEnvironment(
            file_name=env_args['file_path'],
            base_port=env_args['port'],
            no_graphics=not env_args['render']
        )
    if use_GCN:
        env = InfoWrapper_GCN(env)
        env = UnityReturnWrapper_GCN(env)
        env = SamplerWrapper(env, env_args)
    else:
        env = InfoWrapper(env)
        env = UnityReturnWrapper(env)
        env = SamplerWrapper(env, env_args)
    return env
