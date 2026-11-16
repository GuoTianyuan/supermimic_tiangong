"""Small entry points; simulator imports stay inside the execution modules."""
import runpy


def train():
    from supermimic.runners.train import main
    main()


def evaluate():
    runpy.run_module("supermimic.runners.evaluate", run_name="__main__", alter_sys=True)
