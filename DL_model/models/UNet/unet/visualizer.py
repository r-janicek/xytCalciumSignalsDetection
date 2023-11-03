
import numpy as np

__all__ = ["plot_lines", "plot_summary"]


def _fill_to_length(arr, length, value=np.nan):
    return np.concatenate([arr, np.full(length - arr.shape[0], value)])


def plot_lines(vis, xs, ys, **kwargs):
    maxlen = max(len(i) for i in xs)
    
    if maxlen == 0:
        return
    
    xs = [_fill_to_length(i, maxlen) for i in xs]
    ys = [_fill_to_length(i, maxlen) for i in ys]
    
    return vis.line(
        X=np.array(xs).T,
        Y=np.array(ys).T,
        **kwargs
    )


def plot_summary(vis,
                 summary,
                 keys=["training.loss", "testing.loss"],
                 title="Losses",
                 xlabel="Iteration",
                 ylabel="Loss",
                 win=1001):
    
    xs, ys = zip(*[summary.get(i, False) for i in keys])
    return plot_lines(
        vis,
        xs, ys,
        opts={
            'title': title,
            'legend': keys,
            'xlabel': xlabel,
            'ylabel': ylabel
        },
        win=win
    )
