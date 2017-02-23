import matplotlib.pyplot as plt
import numpy as np
pltfmt  = ['b-',  'g-',  'r-',  'c-',  'm-',  'y-',  'k-',  'w-' ]

def normlc(phase, phot, photerr, noecl, bestecl, fignum, j=0, title='Title',
           savedir = './'):
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    
    a = plt.axes([0.15,0.35,0.8,0.55])
    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.4f'))
    plt.title(title)
    plt.errorbar(phase, phot-noecl+1, photerr, fmt='ko',
                 ms=4, lw=1, label='Binned Data')

    plt.plot(phase, bestecl, pltfmt[j], label='Best Fit', lw=2)

    plt.setp(a.get_xticklabels(), visible = False)

    plt.yticks(size=13)
    plt.ylabel('Normalized Flux',size=14)
    
    plt.legend()
    xmin, xmax = plt.xlim()

    plt.axes([0.15,0.1,0.8,0.2])
    flatline = np.zeros(len(phase))
    plt.plot(phase, phot-noecl-bestecl+1, 'ko',ms=4)
    plt.plot(phase, flatline,'k:',lw=1.5)
    plt.xlim(xmin,xmax)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel('Phase',size=14)
    plt.ylabel('Residuals',size=14)


    plt.savefig(savedir + 'normlc.png')
