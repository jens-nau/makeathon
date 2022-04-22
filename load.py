#  This file is detecated to load and process npy files.
import numpy as np
import imageio as iio
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# add name of scan here
slices = np.load('1130.npy')



# print all slices
fig_rows = 5
fig_cols = 5
n_subplots = fig_rows * fig_cols
n_slice = slices.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)


print("n_slices",n_slice)
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(slices[img, :, :], 270), 
                         cmap='gray')
    axs.flat[idx].axis('off')
        
plt.tight_layout()
plt.show()



# #print single slices
# first_slice = slices[0,:,:]
# print(first_slice.shape)
# plt.imshow(first_slice, cmap='gray')
# plt.axis('off')
# plt.show()