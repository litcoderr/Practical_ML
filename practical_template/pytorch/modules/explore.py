# -*- coding: utf-8 -*-
#%%
import modules.dataset as dataset

if __name__ == "__main__":
    config = {
                "X" : ["gender","my_pref","my_rating","partner_rating","interest","guess"],
                "Y" : ["partner"]
                }
    dataset = dataset.SpeedDating_Dataset(config)
    print(dataset.config_)
#%%
    for index, data in enumerate(dataset):
        print("{} | {}".format(index, data))