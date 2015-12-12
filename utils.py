import pprint
from progress.bar import Bar

pp = pprint.PrettyPrinter()

class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'
