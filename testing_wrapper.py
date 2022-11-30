
from runner import Runner
from a_config import Config


def main():
    config = Config()
    runner = Runner(config)

    # runner.test_critical()
    runner.test_manual_scheduler()


if __name__ == '__main__':
    main()
