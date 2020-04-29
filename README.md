# A Probabilistic agent based model of the Germinal Center reaction

<img align="right" src="https://raw.githubusercontent.com/Aurelien-Pelissier/Raman-Imaging-Processing/master/img/Hyperspectral-Image.png" width=400>
Raman microscopy is a very promising technique to increase reliability in clinical diagnosis as it provides information on molecular vibrations and crystal structures, while being non-destructive and requiring minimal sample preparation. It allows high spatial resolution and can provide detailed images of individual cells, for which the extraction of chemical and spatial information of their sub-cellular components have the potential to give a more complete understanding of the underlying biological processes as well as better accuracies in clinical diagnosis. 


&nbsp;



A Raman hyperspectral image consists of a three dimensional matrix, the first two axis corresponding to the pixel position and the third axis being the Raman intensity spectrum at that pixel. One image contains roughly 80000 spectra with 1000 wavenumbers each, in the range of 700 - 3000 cm-1. In addition to experimental artifact (laser focus, sample preparation), Raman data typically involve significant noises, fluorescence background due to water and substrate, as well alteration by cosmic rays. This repository contains the tool to process Hyperspectral Raman images and make it as free as possible from differences in experimental conditions:
       
    Process the hyperspectral Raman image with:  
          - Cosmic ray detection
          - Correction for Irradiance profile
          - Identification and subtraction of Background spectrum
          - Singular Value Decomposition Denoising
          - Polyfit fluorescence correction
          - Spectra area normalization
          - Cell region identification
          - Nucleus and Cytoplasm spectra differentiation
        
        
### Running the code
To process Raman images, run `src/_Main - Raman_analysis.py`. Running the program requires python3, and in addition to standard libraries such as numpy or matplotlib, the program also requires `hdf5storage` (available at https://pypi.org/project/hdf5storage/) to read `.mat` files, and `brokenaxis` (https://github.com/bendichter/brokenaxes) to plot the spectra. Two raw images are provided in `src/Raw_Measurements.py` to show how the code works, but more Raman images are publicly available at https://data.mendeley.com/datasets/dshgffwykw/1


## References

[ref] A. Pelissier, K. Hashimoto, K. Mochizuki, Y. Kumamoto, J. N. Taylor, A. Nakamura, Y. Harada, K. Fujita, T. Komatsuzaki. (2019). Intelligent Measurement Analysis on Single Cell Raman Images for the Diagnosis of Follicular Thyroid Carcinoma [https://arxiv.org/abs/1904.05675]
