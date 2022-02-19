
from runner import Runner
from a_config import Config


def main():
    config = Config()
    runner = Runner(config)

    runner.train_dqn_aleatic_soft()


if __name__ == '__main__':
    main()
