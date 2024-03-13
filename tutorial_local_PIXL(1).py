#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sigma
from sigma.utils import normalisation as norm 
from sigma.utils import visualisation as visual
from sigma.utils.load import SEMDataset, PIXLDataset
from sigma.src.utils import same_seeds
from sigma.src.dim_reduction import Experiment
from sigma.models.autoencoder import AutoEncoder
from sigma.src.segmentation import PixelSegmenter
from sigma.gui import gui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV


# # Load files
# 
# Load the *.bcf* file and create an object of `SEMDataset` (which uses hyperspy as backend.)

# In[232]:


file_path = 'Dourbes_1.hspy'
pixl = PIXLDataset(file_path)
#syn_file_path = 'DataAnorthiteAndAlbite2.csv'
#synthetic = PIXLDataset(syn_file_path)


# # 1. Dataset preprocessing

# ## 1.1 View the dataset
# 
# **Use `gui.view_dataset(sem)`** to check the BSE image, the sum spectrum, and the elemental maps.<br>
# 
# The default *feature list* (or elemental peaks) is extracted from the metadata of the .bcf file. The elemental peaks can be manually modified in `Feature list` and click `set` to set the new feature list. The elemental intensity maps will be generated when clicking the `set` button. In this way, the raw hyperspectral imaging (HSI) dataset has now become elemental maps with dimension of 279 x 514 x 9 (for the test file).
# 
# **How can users determine the elements for analysis?**<br>
# Users can use the interactive widget in the tab "spectra sum spectrum" to check the energy peaks and determine the `Feature list` for further amalyses. 
# 

# In[5]:


gui.view_im_dataset(pixl)
#Na2O_%, MgO_%, Al2O3_%, SiO2_%, P2O5_%, SO3_%, Cl_%, K2O_%, CaO_%, TiO2_%, Cr2O3_%, MnO_%, FeO-T_%, NiO_%, CuO_%, ZnO_%, GeO2_%, Br_%, Rb2O_%, SrO_%, Y2O3_%, ZrO2_%


# In addition to the GUI, we can view the dataset directly with the `sem` object:
# 
# 1. `sem.bse`: access the back-scattered electron (as a hyperspy `Signal2D` object).<br>
# 2. `sem.spectra`: access the spectra dataset (as a hyperspy `EDSSEMSpectrum` object).<br>
# 3. `visual.plot_sum_spectrum(sem.spectra)`: view the sum spectrum (or use hyperspy built-in function `sem.spectra.sum().plot(xray_lines=True)`).<br>
# 4. `sem.feature_list`: view the default chosen elemental peaks in the spectra dataset.<br>
# 5. `sem.set_feature_list`: set new elemental peaks. 

# ## 1.2 Process the dataset

# ### Several (optional) functions to process the dataset:
# 1. `sem.rebin_signal(size=(2,2))`: rebin the spectra signal with the size of 2x2. After rebinning the dataset, we can access the binned spectra or bse data using `sem.spectra_bin` or `sem.bse_bin`. 
# > Note: The size of binning may be changed depending on the number of pixels and signal-to-noise ratio of the EDS spectra. If the input HSI-EDS data contains too many pixels, that may crush the RAM. As for the signal-to-noise, the counts per pixel of the data is better to be higher than 100. In the case of the test dataset, the average counts per pixel is 29.94 before binning and will be 119.76 after binning.
# 
# 2. `sem.peak_intensity_normalisation()`: normalise the x-ray intensity along energy axis.
# 
# 3. `sem.remove_fist_peak(end=0.1)`: remove the first x-ray peak (most likely noise) by calling the function with the argument `end`. For example, if one wants to remove the intensity values from 0-0.1 keV, set `end=0.1`. This is an optional step.
# 
# 4. `visual.plot_intensity_maps`: Plot the elemental intensity maps.

# In[904]:


# Rebin both spectra and bse dataset
#pixl.rebin_signal(size=(2,2))

# normalisation to make the spectrum of each pixel summing to 1.
#pixl.peak_intensity_normalisation()

# Remove the first peak until the energy of 0.1 keV
#pixl.remove_fist_peak(end=0.1) 


# In[905]:


# View the dataset (bse, spectra etc.) again to check differences. 
# Note that a new tab (including the binned elemental maps) will show up only if the user runs the sem.rebin_signal.

#gui.view_pixl_dataset(dataset=pixl)


# The pre-processing steps yield a HSI datacube with the dimension of 139 x 257 x 9 (due to the 2x2 binning).

# ## 1.3 Normalisation
# 
# Before dimensionality reduction, we normalise the elemental maps use `sem.normalisation()`, where we can pass a list containing (optional) sequential normalisation steps.
# 
# > Note that the pre-processing steps are all optional, but if the user opts for the softmax option, z-score  step should be applied beforehand, i.e., the softmax normalization procedure requires input data that is z-score normalized. The purpose of the combinatorial normalization step (z-score + softmax) is to provide the autoencoder with an input that includes global information from all pixels and ‘intentional bias’ within a single pixel. 
# 
# >`neighbour_averaging` is equivilant to apply a 3x3 mean filter to the HSI-EDS data and is an optional step. `zscore` rescales the intensity values within each elemental map such that the mean of all of the values is 0 and the standard deviation is 1 for each map. For example, after z-score normalisation, the Fe Ka map should contain pixels with intensity values that yield `mean=0` and `std=1`. `softmax` is applied within each individual pixel containing z-scores of different elemental peaks. For example, if 5 elemental maps are specified, the 5 corresponding z-scores for each individual pixel will be used to calculate the outputs of softmax.

# In[233]:


# Normalise the dataset using the (optional) sequential three methods.
pixl.normalisation([])


# Use `gui.view_pixel_distributions` to view the intensity distributions after each sequential normalisation process.

# In[126]:


gui.view_pixel_distributions(dataset=pixl, 
                             norm_list=[norm.neighbour_averaging,
                                        norm.zscore,
                                        norm.softmax], 
                             cmap='inferno')


# ## 1.5 Check elemental distribution after normalisation

# In[10]:


print('After normalisation:')
gui.view_intensity_maps(spectra=pixl.normalised_elemental_data, element_list=pixl.feature_list)


# # 2. Dimensionality reduction

# ## 2.1 Method 1: Autoencoder

# ### 2.1.1 Initialise experiment / model

# **Parameters for `Experiment`**<br>
# > `descriptor`: *str*. The name of the model. It will be used as the model name upon saving the model.<br>
# 
# > `general_results_dir`: *path*. The folder path to save the model(the model will automatically save in the specified folder).<br>
# 
# > `model`: *Autoencoder*. The model to be used for training. At this moment, only the vanilla autoencoder can be used. More models (e.g., variational autoencoder) will be implemented in the future versions.<br>
# 
# > `model_args`: *Dict*. Keyword argument for the Autoencoder architecture. The most essential argument is'hidden_layer_sizes' which refers to number of hidden layers and corresponding neurons. For example, if *(512,256,128)*, the encoder will consist of three layers (the first leayer 512 neurons, the second 256 neurons, and the third 128 neurons). The decoder will also be three layers (128 neurons, 256 nwurons, and 512 neurons). The default setting is recommonded in general cases. Increasing the numbers of layers and neurons will increase the complexity of the model, which raise the risk of overfitting.<br>
# 
# > `chosen_dataset`: *np.ndarray*. Normalised HSI-EDS data. The size should be (width, height, number of elemental maps).<br>
# 
# > `save_model_every_epoch`: *bool*. If 'True', the autoencoder model will be saved for each iteration. If 'False', the model will be save only when the loss value is lower than the recorded value.<br>

# In[176]:


# The integer in this function can determine different initialised parameters of model (tuning sudo randomness)
# This can influence the result of dimensionality reduction and change the latent space.
same_seeds(2)

# set the folder path to save the model(the model will automatically save in the specified folder)
result_folder_path='./' 

# Set up the experiment, e.g. determining the model structure, dataset for training etc.
ex = Experiment(descriptor='softmax',
                general_results_dir=result_folder_path,
                model=AutoEncoder,
                model_args={'hidden_layer_sizes':(512,256,128)}, 
                chosen_dataset=pixl.normalised_elemental_data,
                save_model_every_epoch=False)


# ### 2.1.2 Training

# **Parameters for `ex.run_model`**<br>
# > `num_epochs`: *int*. The number of entire passes through the training dataset. 50-100 is recommonded for this value. A rule of thumb is that if the loss value stops reducing, that epoch my be a good point to stop. <br>
# 
# > `batch_size`: *int*. The number of data points per gradient update. Values between 32-128 are recommended. smaller batch size means more updates within an epoch, but is more stochastic for the optimisation process.<br>
# 
# > `learning_rate`: *float* in a range between 0-1. The learning rate controls how quickly the model is adapted to the problem. 1e-4 is recommended. Higher learning rate may yield faster convergence but have a risk to be stuck in an undesirable local minima.<br>
# 
# > `task`: *str*. if 'train_all', all data points will be used for training the autoencoder. If 'train_eval', training will be conducted on the training set (85% dataset) and testing on a testing set (15%) for evaluation. The recommended procedure is to run the 'train_eval' for hyperparameter selection, and 'train_all' for the final analysis.<br>
# 
# > `criterion`: *str*. If 'MSE', the criterion is to measure the mean squared error (squared L2 norm) between each element in the input x and target y. 'MSE' is the only option. Other criteria will be implemented in the future versions.<br>

# In[177]:


# Train the model
ex.run_model(num_epochs=8,
             batch_size=64,
             learning_rate=1e-4, 
             weight_decay=0.5, 
             task='train_all', 
             criterion='MSE'
            ) 
latent = ex.get_latent()


# ### 2.1.3 (Optional) Load pre-trained Autoencoder

# In[13]:


#model_path = './' # model path (the model path should be stored in the folder 'result_folder_path')
#ex.load_trained_model(model_path)
#latent = ex.get_latent()


# ## 2.2 Method 2: UMAP

# In[234]:


from umap import UMAP

# Parameter tuning can be found https://umap-learn.readthedocs.io/en/latest/parameters.html
data = pixl.normalised_elemental_data.reshape(-1,len(pixl.feature_list))
umap = UMAP(
        n_neighbors=25,
        min_dist= 0.05,
        n_components=2,
        metric='euclidean'
    )
latent = umap.fit_transform(data)


# In[208]:


from sklearn.manifold import TSNE
data = pixl.normalised_elemental_data.reshape(-1,len(pixl.feature_list))
latent = TSNE(n_components=2,
              learning_rate='auto',
              init='random',
              perplexity=40,
              #early_exaggeration=,
              n_iter=5000).fit_transform(data)
latent.shape


# In[226]:


from sklearn.manifold import MDS
data = pixl.normalised_elemental_data.reshape(-1,len(pixl.feature_list))

embedding = MDS(n_components=2, normalized_stress='auto', eps=1)
latent = embedding.fit_transform(data)
latent.shape



# In[128]:


#from sklearn.manifold import Isomap
#
#embedding = Isomap(n_components=2)
#latent = embedding.fit_transform(pixl)
#latent.shape


# # 3. Pixel segmentation: 

# ## 3.1 Method 1: Gaussian mixture modelling (GMM) clustering 

# ### 3.1.1 Measure Baysian information criterion (BIC)

# The `gui.view_bic` iteratively calculates the BIC for Gaussian mixture models using the number of Gaussian components.
# 
# **Parameters for `gui.view_bic`**<br>
# > `latent`: *np.ndarray*. 2D representations learned by the autoencoder. The size of the input array must be (n, 2), where n is number of total data points.<br>
# 
# > `model`: *str*. Model for calculation of BIC. Only 'GaussianMixture' is available for now.<br>
# 
# > `n_components`: *int*. If `n_components=20`, it shows the BIC values for GMM using `n_components` from 1 to 20.<br>
# 
# > `model_args`: *Dict*. Keyword arguments for the GMM model in sklearn. For example, `random_state` is to specify the random seed for optimisation (This can make the results reproduciable); `init_params` is to specify the parameter initialisation of the GMM model ('kmeans' is recommended). See mode detail [here](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).<br>

# In[169]:


gui.view_bic(latent=latent,
             model='GaussianMixture',
             n_components=20,
             model_args={'random_state':6, 'init_params':'kmeans'})


# ### 3.1.2 Run GMM

# **Parameters for `PixelSegmenter`**<br>
# > `latent`: *np.ndarray*. The size of the input array must be (n, 2), where n is number of total data points.<br>
# 
# > `dataset_norm`: *np.ndarray*. Normalised HSI-EDS data. The size should be *(width, height, number of elemental maps)*. <br>
# 
# > `sem`: *SEMDataset*. The SEM object created in the beginning.<br>
# 
# > `method`: *str*. Model for clustering.<br>
# 
# > `method_args`: *Dict*. Keyword arguments for the GMM model in sklearn. [See mode detail here](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).<br>

# In[237]:


ps = PixelSegmenter(latent=latent, 
                    dataset=pixl,
                    method="GaussianMixture",
                    method_args={'n_components':8, 'random_state':1, 'init_params':'kmeans'} )
 # can change random_state to different integer i.e. 10 or 0 to adjust the clustering result.


# ## 3.2 Method 2: HDBSCAN clustering

# In[142]:


# hyperparameter tuning can be found https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#hyperparameter-robustness
ps = PixelSegmenter(latent=latent, 
                    dataset=pixl,
                    method="HDBSCAN",
                    method_args=dict(min_cluster_size=25, min_samples=20,
                                     max_cluster_size=int(len(latent)/10),
                                     cluster_selection_epsilon=4e-1) )
print(ps.model.labels_.max())
ps


# ## 3.3 Method 3:  AffinityPropagation clustering

# In[146]:


ps = PixelSegmenter(latent=latent, 
                    dataset=pixl,
                    method="AffinityPropagation",
        
                    method_args=dict(preference = -1000,
                                     max_iter = 500,
                                     damping = 0.95,
                                    random_state=0))
print(ps.model.labels_.max())
ps


# ## 3.4 Method 4:  MeanShift clustering

# In[149]:


ps = PixelSegmenter(latent=latent, 
                    dataset=pixl,
                    method="MeanShift",
                    method_args=dict(bandwidth =1.9
                                    ))
print(ps.model.labels_.max()+1)
ps


# In[89]:


ps.labels.shape


# ## 3.3 Visualisation

# ### 3.3.1 Checking latent space

# **Parameters for `gui.view_latent_space`**<br>
# > `ps`: *PixelSegmenter*. The object of PixelSegmetor which is just created.<br>
# 
# > `color`: *bool*. If `True`, the the latent space will be colour-coded based on their cluster labels.<br>

# In[346]:


rawdata = pd.read_csv('Data/Alfalfa.csv', header=1)
rawdata.shape


# In[349]:


#Creates a pattern for the cluster labels to be stored in
#change raw_data_size and raw_data_range for different datasets
raw_data_size = 3249
raw_data_range = 57
pattern = np.arange(0, raw_data_range) + np.arange(0, raw_data_size, raw_data_range)[:, np.newaxis]
result1 = ps.labels.reshape((-1, raw_data_range)).T.ravel()[pattern]
result = result1.flatten()
reshaped = result.reshape((raw_data_size,1))


# In[350]:


#setting up variables for later code, reads in rawdata and creates the appended data, apdat
header = np.array(['Clusters',])
apdata = np.hstack((header, result))


# In[351]:


#Code for concatenating the cluster labels with the raw data file, messy but works
updated_data = np.concatenate((rawdata, reshaped), axis=1)
updated_data
headers = updated_data[0]
header_list = list(headers)


# In[356]:


#change plotted_cluster to change cluster
plotted_cluster = 4
x_data = updated_data[:, 12].astype(float)
y_data = updated_data[:, 13].astype(float)
clusters = updated_data[1:, 55].astype(float)
indexedcluster = np.where(clusters == plotted_cluster)
plt.scatter(x_data, y_data, label='All Data', color='blue', s=4)
plt.scatter(x_data[indexedcluster], y_data[indexedcluster], label='Fe-Si material', color='red', s=4)

plt.ylim(0,70)
plt.xlabel('MnO%')
plt.ylabel('FeO%')

plt.legend()
plt.savefig('FeMnA.png')
plt.show()


# In[75]:


#Plotting K+Na vs Fe
sum_column = updated_data[:,8] + updated_data[:,1] + updated_data[:,9]
array_with_sum_column = np.column_stack((updated_data, sum_column))
Si_data = array_with_sum_column[:, 4].astype(float)
NaCaK_data = array_with_sum_column[:, 56].astype(float)
plt.scatter(Si_data, NaCaK_data, label='Plot', color='red', s=4)
NaCaKmean = np.mean(NaCaK_data)
Simean = np.mean(Si_data)
plt.scatter(Simean, NaCaKmean, color='blue', s=10)
#plt.show()


# In[160]:


# Plot latent space (2-dimensional) with corresponding Gaussian models
gui.view_latent_space(ps=ps, color=True)


# **Parameters for `gui.check_latent_space`**<br>
# > `ps`: *PixelSegmenter*. The object of PixelSegmetor which is just created.<br>
# 
# > `ratio_to_be_shown`: *float*. The value must be between 0-1. For example, if 0.5, the latent space will only show 50% data points.<br>
# 
# > `show_map`: *bool*. If `True`, the corresponding locations of the data points will be overlaid on the BSE image.<br>
# 

# **Parameters for `gui.check_latent_space`**<br>
# > `ps`: *PixelSegmenter*. The object of PixelSegmetor which is just created.<br>
# 
# > `bins`: *int*. Number of bins for the given interval of the latent space.<br>

# In[238]:


# visualise the latent space
gui.check_latent_space(ps=ps,ratio_to_be_shown=1.0, show_map=True)


# In[342]:


# check the density of latent space
#gui.plot_latent_density(ps=ps, bins=50)


# ### 3.3.2 Checking each clusters

# In[415]:


gui.show_cluster_distribution(ps=ps)


# ### 3.3.3 Checking cluster map

# In[182]:


# Plot phase map using the corresponding GM model
gui.view_phase_map(ps=ps, alpha_cluster_map=1)


# **Parameters for `gui.view_clusters_sum_spectra`**<br>
# 
# > `ps`: *PixelSegmenter*. The object of PixelSegmetor which is just created.<br>
# 
# > `normalisation`: *bool*. If `True`, the sum spectra will be normalised.<br>
# 
# > `spectra_range`: *tuple*. Set the limits of the energy axes (in *KeV*).<br>

# In[592]:


gui.view_clusters_sum_spectra(ps=ps, normalisation=True, spectra_range=(0,8))


# # 4. Unmixing cluster spectrums using Non-negative Matrix Fatorization (NMF)

# **Parameters for `gui.get_unmixed_spectra_profile`**<br>
# > `clusters_to_be_calculated`: *str* or *List*. If `'All'`, all cluster-spectra will be included in the data matrix for NMF. If it is a list of integers (e.g. [0,1,2,3]), only #0, #1, #2, and #3 cluster-spectra will be used as the data matrix for NMF.<br>
# 
# > `normalised`: *bool*. If `True`, the sum spectra will be normalised before NMF unmixing.<br>
# 
# > `method`: *str*. Model to be used.<br>
# 
# > `method_args`: *Dict*. Keyword arugments for the NMF method (or others). [See more detail here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).<br>

# In[275]:


weights, components = ps.get_unmixed_spectra_profile(clusters_to_be_calculated='All', 
                                                 n_components=7,
                                                 normalised=False, 
                                                 method='NMF', 
                                                 method_args={'init':'nndsvd'})


# In[33]:


weights, components = ps.get_unmixed_spectra_profile_2(clusters_to_be_calculated='All', 
                                                 n_components=3,
                                                 normalised=False, 
                                                 method='PCA', 
                                                 method_args={})


# In[34]:


weights, components = ps.get_unmixed_spectra_profile_3(clusters_to_be_calculated='All', 
                                                 n_components=3,
                                                 normalised=False, 
                                                 method='SPCA',
                                                 
                                                 method_args={})


# In[44]:


weights, components = ps.get_unmixed_spectra_profile_4(clusters_to_be_calculated='All', 
                                                 n_components=6,
                                                 normalised=False, 
                                                 method='FICA',
                                                 
                                                 method_args={})


# In[29]:


#weights, components = ps.get_unmixed_spectra_profile_5(clusters_to_be_calculated='All', 
#                                                       n_components=6,
#                                                       normalised=False, 
#                                                       method='CCA',
#                                                 
#                                                 method_args={})
#still not working


# In[358]:


#creating variable clusterdf, a df of all the points in a cluster
#change cluster_of_interest to change cluster analysed 
cluster_of_interest = 4
clusters_column = updated_data[:, 55]
dfuseful = updated_data[clusters_column == cluster_of_interest]
dfuseful = pd.DataFrame(dfuseful)
dfuseful = dfuseful.iloc[:, 1:14]
#change dictionary for different dictionaries
dictionary = pd.read_csv('Data/Mindata2.csv', header=0).iloc[: , 1:14]


# In[359]:


#Code for using the OMP algorithim to sort the data into raw mineral components based on a dictionary
from sklearn.decomposition import SparseCoder
#starred code below is for bulk composition analysis
 #df3 = pd.read_csv('Data/130744834 - Alfalfa.csv', header =1)
 #dfuseful = df3.iloc[:, 1:23]

dfuseful = dfuseful.astype(float)
dictionary = dictionary.astype(float) 
coder = SparseCoder(
    dictionary=dictionary, transform_algorithm='omp',
    transform_alpha=0.15,
    transform_n_nonzero_coefs=14
)
transformed = coder.transform(dfuseful)
#output is transformed variable, stored as a np array


# In[360]:


#extra bit of code that shows a random set of 20 points from the OMP transformed dataset
tabular = pd.DataFrame(transformed)
ran = np.random.randint(1,50)
selected = tabular.iloc[ran:(ran+20)]


# In[361]:


#code to average across all rows, normalisation and then plotting as bar graph
row_means = tabular.mean(axis=0)
dictionary1 = pd.read_csv('Data/Mindata2.csv', header=0)
plt.xticks(range(len(row_means)), dictionary1['Weight Percent References'], rotation=90)
plt.plot(row_means.T, 'r+')
plt.gcf().subplots_adjust(bottom=0.5)
plt.savefig('FeSiOMPA.png')
plt.show()


# In[276]:


gui.show_unmixed_weights_and_compoments(ps=ps, weights=weights, components=components)


# ## Check abundance map for components (using RGB maps)

# In[13]:


gui.show_abundance_map(ps=ps, weights=weights, components=components)


# ## Statistics info from clusters

# In[18]:


gui.show_cluster_stats(ps=ps)


# In[28]:


pd.read_csv(updated_data)


# In[367]:


#histogram
alldata = updated_data[:, 13]
clusterdata = np.where(updated_data[:, 55] == 4)
selecteddata = alldata[clusterdata]
plt.hist(alldata, bins=50, color='skyblue', edgecolor='black')
plt.hist(selecteddata, bins=50, color='red', edgecolor='black')
plt.xlabel('Fe content')
plt.ylabel('Frequency')
plt.xlim(0,100)
plt.ylim(0,500)
plt.savefig('FeSiHistA')
plt.show


# In[ ]:




