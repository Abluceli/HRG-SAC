
# coding: utf-8
"""
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -i,--inference              推断 [default: False]
    -a,--algorithm=<name>       算法 [default: ppo]
    -c,--config-file=<file>     指定模型的超参数config文件 [default: None]
    -e,--env=<file>             指定环境名称 [default: None]
    -p,--port=<n>               端口 [default: 5005]
    -u,--unity                  是否使用unity客户端 [default: False]
    -g,--graphic                是否显示图形界面 [default: False]
    -n,--name=<name>            训练的名字 [default: None]
    -s,--save-frequency=<n>     保存频率 [default: None]
    -m,--models=<n>             同时训练多少个模型 [default: 1]
    --store-dir=<file>          指定要保存模型、日志、数据的文件夹路径 [default: None]
    --seed=<n>                  指定模型的随机种子 [default: 0]
    --max-step=<n>              每回合最大步长 [default: None]
    --max-episode=<n>           总的训练回合数 [default: None]
    --sampler=<file>            指定随机采样器的文件路径 [default: None]
    --load=<name>               指定载入model的训练名称 [default: None]
    --fill-in                   指定是否预填充经验池至batch_size [default: False]
    --prefill-choose            指定no_op操作时随机选择动作，或者置0 [default: False]
    --gym                       是否使用gym训练环境 [default: False]
    --gym-agents=<n>            指定并行训练的数量 [default: 1]
    --gym-env=<name>            指定gym环境的名字 [default: CartPole-v0]
    --gym-env-seed=<n>          指定gym环境的随机种子 [default: 0]
    --render-episode=<n>        指定gym环境从何时开始渲染 [default: None]
    --info=<str>                抒写该训练的描述，用双引号包裹 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --max-episode 1000 --sampler C:/test_sampler.yaml
    python run.py -a ppo -u -n train_in_unity --load last_train_name
    python run.py -ui -a td3 -n inference_in_unity
    python run.py -gi -a dddqn -n inference_with_build -e my_executable_file.exe
    python run.py --gym -a ppo -n train_using_gym --gym-env MountainCar-v0 --render-episode 1000 --gym-agents 4
    python run.py -u -a ddpg -n pre_fill --fill-in --prefill-choose
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import sys
import time
NAME = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
import platform
BASE_DIR = f'C:/RLData' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData'

from typing import Dict
from copy import deepcopy
from docopt import docopt
from multiprocessing import Process
from common.agent import Agent
from common.yaml_ops import load_yaml
from common.config import Config


def get_options(options: Dict):
    f = lambda k, t: None if options[k] == 'None' else t(options[k])
    op = Config()
    op.add_dict(dict([
        ['inference',       bool(options['--inference'])],
        ['algo',            str(options['--algorithm'])],
        ['algo_config',     f('--config-file', str)],
        ['env',             f('--env', str)],
        ['port',            int(options['--port'])],
        ['unity',           bool(options['--unity'])],
        ['graphic',         bool(options['--graphic'])],
        ['name',            f('--name', str)],
        ['save_frequency',  f('--save-frequency', int)],
        ['models',          int(options['--models'])],
        ['store_dir',       f('--store-dir', str)],
        ['seed',            int(options['--seed'])],
        ['max_step',        f('--max-step', int)],
        ['max_episode',     f('--max-episode', int)],
        ['sampler',         f('--sampler', str)],
        ['load',            f('--load', str)],
        ['fill_in',         bool(options['--fill-in'])],
        ['prefill_choose',  bool(options['--prefill-choose'])],
        ['gym',             bool(options['--gym'])],
        ['gym_agents',      int(options['--gym-agents'])],
        ['gym_env',         str(options['--gym-env'])],
        ['gym_env_seed',    int(options['--gym-env-seed'])],
        ['render_episode',  f('--render-episode', int)],
        ['info',            f('--info', str)]
    ]))
    return op
        

def agent_run(*args):
    Agent(*args)()


def run():
    if sys.platform.startswith('win'):
        import win32api
        import win32con
        import _thread

        def _win_handler(event, hook_sigint=_thread.interrupt_main):
            if event == 0:
                hook_sigint()
                return 1
            return 0
        # Add the _win_handler function to the windows console's handler function list
        win32api.SetConsoleCtrlHandler(_win_handler, 1)

    options = docopt(__doc__)
    options = get_options(dict(options))
    print(options)

    default_config = load_yaml(f'./config.yaml')
    # gym > unity > unity_env
    model_args = Config(**default_config['model'])
    train_args = Config(**default_config['train'])
    env_args = Config()
    buffer_args = Config(**default_config['buffer'])
    
    model_args.algo = options.algo
    model_args.algo_config = options.algo_config
    model_args.seed = options.seed
    model_args.load = options.load

    if options.gym:
        train_args.add_dict(default_config['gym']['train'])
        train_args.update({'render_episode': options.render_episode})
        env_args.add_dict(default_config['gym']['env'])
        env_args.type = 'gym'
        env_args.env_name = options.gym_env
        env_args.env_num = options.gym_agents
        env_args.env_seed = options.gym_env_seed
    else:
        train_args.add_dict(default_config['unity']['train'])
        env_args.add_dict(default_config['unity']['env'])
        env_args.type = 'unity'
        env_args.port = options.port
        env_args.sampler_path = options.sampler
        if options.unity:
            env_args.file_path = None
            env_args.env_name = 'unity'
        else:
            env_args.update({'file_path': options.env})
            if os.path.exists(env_args.file_path):
                env_args.env_name = os.path.join(
                    *os.path.split(env_args.file_path)[0].replace('\\', '/').replace(r'//', r'/').split('/')[-2:]
                )
            else:
                raise Exception('can not find this file.')
        if options.inference:
            env_args.train_mode = False
            env_args.render = True
        else:
            env_args.train_mode = True
            env_args.render = options.graphic

    train_args.index = 0
    train_args.name = NAME
    train_args.inference = options.inference
    train_args.fill_in = options.fill_in
    train_args.prefill_choose= options.prefill_choose
    train_args.base_dir = os.path.join(options.store_dir or BASE_DIR, env_args.env_name, model_args.algo)
    train_args.update(
        dict([
            ['name', options.name],
            ['max_step', options.max_step],
            ['max_episode', options.max_episode],
            ['save_frequency', options.save_frequency],
            ['info', options.info]
        ])
    )

    if options.inference:
        Agent(env_args, model_args, buffer_args, train_args).evaluate()

    trails = options.models
    if trails == 1:
        agent_run(env_args, model_args, buffer_args, train_args)
    elif trails > 1:
        processes = []
        for i in range(trails):
            _env_args = deepcopy(env_args)
            _model_args = deepcopy(model_args)
            _model_args.seed += i * 10
            _buffer_args = deepcopy(buffer_args)
            _train_args = deepcopy(train_args)
            _train_args.index = i
            if _env_args.type == 'unity':
                _env_args.port = env_args.port + i
            p = Process(target=agent_run, args=(_env_args, _model_args, _buffer_args, _train_args))
            p.start()
            time.sleep(10)
            processes.append(p)
        [p.join() for p in processes]
    else:
        raise Exception('trials must be greater than 0.')


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        sys.exit()
