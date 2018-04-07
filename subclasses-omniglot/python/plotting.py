import matplotlib.pyplot as plt


def _set_params(cols, rows):
    num_splits = None
    if rows is not None:
        num_splits = rows
        cols = None
    if cols is not None:
        num_splits = cols
        rows = None
    return cols, num_splits, rows


def plots(ims, figsize=(16, 8), rows=None, cols=None, interp=False, titles=None):
    plt.figure(figsize=figsize)
    cols, num_splits, rows = _set_params(cols, rows)

    tmp_num_splits = len(ims) // num_splits
    if tmp_num_splits * num_splits < len(ims):
        tmp_num_splits += 1

    for i in range(len(ims)):

        if rows is not None:
            sp = plt.subplot(num_splits, tmp_num_splits, i + 1)
        if cols is not None:
            sp = plt.subplot(tmp_num_splits, num_splits, i + 1)
        sp.axis('Off')

        if titles is not None:
            sp.set_title(titles[i], fontsize=16)

        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()
