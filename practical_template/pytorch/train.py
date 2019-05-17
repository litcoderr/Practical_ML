import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# My Modules yay!!!!
import modules.dataset as dataset
import modules.model as models
#%%
## Constants used for training
## Hyper Parameters
batch_size = 64
n_cpu = 2
n_gpu = 0
lr = 0.001
n_epochs = 1000

config_ = {
    "X" : ["gender","my_pref","my_rating","partner_rating","interest","guess"],
    "Y" : ["partner"]
    }

config = {
    "X" : ["gender","my_pref","my_rating","partner_rating","interest","guess"],
    "Y" : ["partner"]
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# load dataset to train loader
train_dataset = dataset.SpeedDating_Dataset(config, "train")
train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=n_cpu,shuffle=True,drop_last=True)
len_iteration = len(train_loader)

test_dataset = dataset.SpeedDating_Dataset(config, "test")
test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), num_workers=n_cpu,shuffle=False,drop_last=False)

sample_x, sample_y = train_dataset[0]
input_size, output_size = int(sample_x.shape[0]), int(sample_y.shape[0])

# load my model
model = models.V1(input_size, output_size).to(device)

# define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

# Make training Sequence
def train(x,y):
    # initialize optimizer
    optimizer.zero_grad()
    
    # forward
    y_ = model(x)
    
    # calculate loss
    loss = criterion(y_, y)
    
    # back propagate
    loss.backward()
    
    # take step
    optimizer.step()
    
    return loss
    
    
#%%
if __name__ == "__main__":
    # Start training
    train_rec = [] # record of losses
    valid_rec = [] # record of validation losses
    for epoch in range(n_epochs):
        iteration = 0
        # train
        for batch_idx, (X,Y) in enumerate(train_loader):
            iteration += 1
            train_loss = train(X,Y) # train and outputs train loss
            
        # run validation
        for batch_idx, (X,Y) in enumerate(test_loader):
            y_ = model(X)
            valid_loss = criterion(y_, Y)
            
        print("epoch [{}/{}] {:.2f}% loss: {} valid_loss: {}".format(epoch+1, n_epochs, (iteration/len_iteration)*100, train_loss, valid_loss))
        train_rec.append(train_loss)
        valid_rec.append(valid_loss)
#%%
    plt.plot(train_rec)
    #plt.ylim(0,1)
    plt.title("train loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    
    plt.plot(valid_rec)
    #plt.ylim(0,1)
    plt.title("validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    
    