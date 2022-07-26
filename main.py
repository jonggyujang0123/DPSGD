"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import *

from agents import *
class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'mode',
        metavar='mode',
        default='None',
        help='mode, choose "train" or "test"')
    arg_parser.add_argument(
        'config',
        metavar='config_yaml_file',
        default='None',
        help='The Configuration file in yaml format')
    arg_parser.add_argument(
        'gpu_id',
        metavar='gpu_id',
        default='0',
        nargs='?',
        action=SplitArgs,
        help='gpu_id')
    args = arg_parser.parse_args()

    # parse the config json file
    config, config_dict = process_config(args.config)
    #gpu_ids = args.gpu_id
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config, config_dict, args.gpu_id, args.mode)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
