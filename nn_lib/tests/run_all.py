import unittest


def run_all():
    """
    Run all the tests
    """
    loader = unittest.TestLoader()
    start_dir = './'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    run_all()
