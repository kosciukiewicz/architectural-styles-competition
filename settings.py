import os

PATH_TO_DATA = None
PATH_TO_OUTPUT = None

PROJECT_PATH = os.path.dirname(__file__)


try:
    from user_settings import *  # silence pyflakes
except ImportError:
    pass
