from __init__ import array_lib as np
from plt import plt


def mse(f, f_ref, normalize=False, db=False):
    ms = np.sum((f-f_ref)**2)

    ms /= np.sum(f_ref**2) if normalize else f_ref.size
    if db: ms = 10.*np.log10(ms)

    return ms


def rmse(f, f_ref, normalize=False, db=False):
    ms = mse(f, f_ref, normalize=normalize)
    return 5.*np.log10(ms) if db else np.sqrt(ms)


def psnr(f, f_ref):
    rmin = np.min(f_ref)
    range = np.max(f_ref)-rmin*(rmin<0)

    return 10.*np.log10( mse(f, f_ref)**2 / range)


def mutual_info(f, f_ref, bins=100):
    from scipy.stats import entropy #cupy-compatible but not?

    hist_values, _ = np.histogramdd([f.ravel(), f_ref.ravel()],
                                    bins=bins, density=True)

    return ( entropy(np.sum(hist_values, axis=0).get()) + entropy(np.sum(hist_values, axis=1).get()) ) \
           / entropy(hist_values.ravel().get())


def rescale_and_compare(f, f_ref, compare, rescale=None, **kwargs):
    if rescale is None: rescale = lambda x: x/np.sum(x)

    nf = rescale(f)
    nf_ref = rescale(f_ref)

    return compare(nf, nf_ref, **kwargs)


def compare_plots(f, f_ref, compare=None, rescale=None, single_cb=True):

    if rescale is None: rescale = lambda x: x
    if compare is None: compare = lambda x,y : np.abs(x-y)

    if not isinstance(f, list): f = list(f)

    ncols = 1+len(f)
    nrows = 2
    magni = 5
    fig = plt.figure(figsize=(magni*ncols, magni*nrows), constrained_layout=True)
    

    images = [rescale(im) for im in ([f_ref] + f)]
    diffs = [compare(im, images[0]) for im in images]

    images = np.array(images)
    vmin = np.min(images).item(); vmax = np.max(images).item()

    diffs = np.array(diffs)
    vmind, vmaxd = (vmin, vmax) if single_cb else (np.min(diffs).item(), np.max(diffs).item())
    
    for index in range(1, ncols+1):
        fig.add_subplot(nrows, ncols, index)
        im = plt.imshow(images[index-1], vmin=vmin, vmax=vmax)

        fig.add_subplot(nrows, ncols, index+ncols)
        df = plt.imshow(diffs[index-1], vmin=vmind, vmax=vmaxd)
    
    axs = fig.axes

    sz = nrows*ncols
    if single_cb:
        fig.colorbar(im, ax=[axs[sz-2], axs[sz-1]], location="right", fraction=0.05)# pad=0.04)
    else:
        fig.colorbar(im, ax=axs[sz-2], fraction=0.05) #[axs[0], axs[1]]
        fig.colorbar(df, ax=axs[sz-1], fraction=0.05)

    for ax in axs: ax.axis('off')
    plt.show()

    return fig