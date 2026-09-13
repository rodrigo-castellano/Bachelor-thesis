import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from numpy import save
import random


# Load the files and put them as dataframe
def plot_scatter(lista, names, title, xlabel, ylabel):
    '''Function that shows/saves a scatter plot
    Args: lista: data to plot, names: names of the particles, title: title of the plot, xlabel,ylabel: name of labels'''
    plt.title(title)
    contador = 1
    for i, j in zip(lista, names):
        plt.scatter(contador, i[i[4] == 1][3].values.max(), label=j)
        contador = contador + 1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")

    plt.savefig("images/plots/MaxInt.png")
    plt.show()

    return None

def plot_histogram (lista, names, title, xlabel, ylabel, bins , limit, xlim, ylim, pixel):
    '''Function that shows/saves a histogram1
        Args: lista: data to plot, names: names of the particles, title: title of the plot, xlabel,ylabel: name of labels
               bins: number of bins, limit: amount of data to plot, pixel: whether activated pixels or not,
               xlim,ylim: limits in x and y axes'''

    for i, j in zip(lista, names):
        plt.hist(i.loc[i[4] == pixel][3][:limit], bins=bins, label=j)

    plt.title(title)
    plt.legend(loc="upper right")
    plt.ylim(0, ylim)
    plt.xlim(0, xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title=title.replace(" ", "_")
    title = title.replace(":", "_")
    plt.savefig("images/plots/" + title + ".png")
    plt.show()
    return None

def plot_histogram2(df, title, xlabel, ylabel,particle, bins , limit, xlim, ylim, pixel, xmin=0, ymin=0):
    '''Same as plot_histogram, but this is used to compare particles'''

    plt.hist(df.loc[df[4] == pixel][3][:limit], bins=bins, color='c', label=particle)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.ylim(ymin, ylim)
    plt.xlim(xmin, xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    title = title.replace(" ", "_")
    title = title.replace(":", "_")
    plt.savefig("images/plots/" + title + ".png")
    plt.show()
    return None

def plot(lista, names):
    '''Function used to plot and save plots'''

    for i in range(7):
        lista[i] = pd.DataFrame(data=lista[i])

    title = 'Maximum of intensity for activated pixels'
    xlabel = 'Particle'
    ylabel = 'Maximum of intensity'
    #plot_scatter(lista, names, title, xlabel, ylabel)

    title = 'Histogram: deactivated pixels in the range [0,0.01]'
    xlabel = 'Intensity'
    ylabel = 'Frequency'
    plot_histogram(lista, names, title, xlabel, ylabel, bins=1000, limit=3500000, xlim=0.01, ylim=200000, pixel=0)

    title = 'Histogram: activated pixels in the range [0,1]'
    plot_histogram(lista, names, title, xlabel, ylabel, bins=1000, limit=7500, xlim=1, ylim=140, pixel=1)

    df1 = lista[0]
    df2 = lista[1]
    df1 = df1.sample(frac=1).reset_index(
        drop=True)  # barajo gamma para que no haya problema al coger una cierta cantidad de datos de gamma

    title = 'Histogram: Activated pixels (gamma)'
    plot_histogram2(df1, title, xlabel, ylabel,particle='gamma', bins=500 , limit=100818, xlim=1, ylim=4000, pixel=1)

    title = 'Histogram: Activated pixels (electron)'
    plot_histogram2(df2, title, xlabel, ylabel,particle='electron', bins=500 , limit=100818, xlim=1, ylim=4000, pixel=1)

    title = 'Histogram: Activated pixels (gamma)'
    plot_histogram2(df1, title, xlabel, ylabel, particle='gamma', bins=500 , limit=100818, xlim=1, ylim=15, pixel=1, xmin=0.4)

    title = 'Histogram: Activated pixels (electron)'
    plot_histogram2(df2, title, xlabel, ylabel, particle='electron', bins=500 , limit=100818, xlim=1, ylim=15, pixel=1, xmin=0.4)









