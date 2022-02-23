
from runner import Runner
from a_config import Config


def main():
    config = Config()
    runner = Runner(config)

    # runner.train_dqn_aleatoric()
    # runner.train_dqn_aleatoric_soft()
    # runner.train_dqn_deterministic()

    runner.train_try_teach_critical()


if __name__ == '__main__':
    main()
