# Part 2 — Cherenkov Telescope Array images
In this project cosmic rays from the Cherenkov Telescope Array are analyzed.

Features

     •X position in the hexagonal grid
     •Y position in the hexagonal grid
     •Number of event
     •Intensity of the pixel in (X,Y)

Labels: 

    0 --> gamma  
    
    1 --> electron
    
    2 --> proton 
    
    3 --> nitrogen  
    
    4 --> iron 
    
    5 --> silicum
    
    6 --> helium
     
Algotirthms that are used: Neural network and convolutional neural network

## The simulation data is not here

Seven zips, one per primary particle, **674 MB** — with `gamma.zip` alone at
297 MB. GitHub refuses any single file over 100 MB, so they cannot live in
this repository. They are archived instead, at

    ARCHIVE\data\projects\ugr-tfg\part-2-cta\simulation-data\

and that copy is the only one. Unlike the Part 1 data, there is no public
source to re-download them from.

`CTA/dataset.py` turns them into numpy arrays with `np.save('data/npy/…')` —
`*_n.npy`, `*_np2.npy`, `*_image.npy` and `gamma_image_moved.npy`. Those come
to about 4.7 GB and are **not** kept anywhere: they are regenerated from the
zips by the code in this folder.

## results/

The measured output of the runs, kept because the searches are expensive.

    grid-search/
      grid-search-1.csv              12 configurations over dropout_rate,
                                     init_mode and learn_rate. The recorded
                                     mean_fit_time is ~2800 s per fit, two CV
                                     splits each — roughly 19 hours of compute,
                                     in 3 KB
      general-classification.csv     32 configurations over conv, dropout_rate,
                                     kernel, pool and stride. Read back by
                                     general-classification.ipynb and its v2
    history/
      history.csv                    read back by six of the notebooks
      history-gamma-vs-all.csv       the strong binary result, 0.95 -> 0.98
      history-general-classification.csv
      colab-training-history.csv     the longest run, 50 epochs, from Colab
    cnn-history/
      history.csv, accuracy.png, loss.png     the CNN run
    plots/
      histogram*.png, max.png        activated-pixel distributions

The notebooks write these with `to_csv` and read them back afterwards, so a
training cell can be skipped and the saved run used instead. They use paths
relative to their own directory (`history/history.csv`), so copy what you need
beside the notebook before re-running it.

`history-2.csv` used to sit next to `history.csv`. It was byte-identical to
it, so it was dropped rather than migrated.
