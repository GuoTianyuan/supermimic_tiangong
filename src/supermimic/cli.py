"""Small entry points; simulator imports stay inside the execution modules."""
import runpy


def train():
    runpy.run_module("supermimic.runners.train", run_name="__main__", alter_sys=True)


def evaluate():
    runpy.run_module("supermimic.runners.evaluate", run_name="__main__", alter_sys=True)
