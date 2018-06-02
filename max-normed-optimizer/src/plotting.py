import matplotlib


def set_display_settings():
    """Set bigger font size"""
    font = {
        'family': 'sans-serif',
        'weight': 'normal',
        'size': 18
    }

    matplotlib.rc('font', **font)