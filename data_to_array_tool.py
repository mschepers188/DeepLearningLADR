import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

class DataLoader:
    def __init__(self, f_paths):
        self.paths = f_paths
        print(f'Importing images from: {self.paths}')

    def imgs_to_array(self):
        """
        Takes self.file and grabs n images in folder
        """
        array_list = []
        for i in self.paths:
            filelist = glob.glob(i)
            x_tmp = np.array([np.array(ImageOps.grayscale(Image.open(fname))) for fname in filelist])/255
            # x_tmp_reshaped = []
            array_list.append(x_tmp)

        # print(array_list)
        # print(len(array_list))
        return array_list

    def array_reshaper(self, array_list):
        """
        Reshape arrays items to one dimension to become a row entries
        """
        array_list_reshaped = []
        for arr in array_list:
            array_list_reshaped.append(arr.reshape(arr.shape[0], -1))
            # print(arr.shape[0])
        return array_list_reshaped

    def arrays_to_df(self, array_list):
        """
        Converts array with 1 dimensional items to a dataframe
        """
        arrays_tmp = []
        for n, i in enumerate(array_list):
            df = pd.DataFrame(i).assign(Label=n)
            arrays_tmp.append(df)
        df_conc = pd.concat(arrays_tmp)
        return df_conc

if __name__ == "__main__":
    path_list = [
        'C:\\Users\\Martin Schepers\\Documents\\dataset2-master\\dataset2-master\\images\\TRAIN\\EOSINOPHIL\\*.jpeg',
        'C:\\Users\\Martin Schepers\\Documents\\dataset2-master\\dataset2-master\\images\\TRAIN\\LYMPHOCYTE\\*.jpeg',
        'C:\\Users\\Martin Schepers\\Documents\\dataset2-master\\dataset2-master\\images\\TRAIN\\MONOCYTE\\*.jpeg',
        'C:\\Users\\Martin Schepers\\Documents\\dataset2-master\\dataset2-master\\images\\TRAIN\\NEUTROPHIL\\*.jpeg'
    ]

    dataloader = DataLoader(path_list)
    arrays = dataloader.imgs_to_array()
    arrays = dataloader.array_reshaper(arrays)
    for i in range(4):
        print(arrays[i].shape)
    array_df = dataloader.arrays_to_df(arrays)
    print(array_df.head())