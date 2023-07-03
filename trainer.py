import torch 
from torch import nn, optim
from torch.utils.data import DataLoader



class DataHandler() :
    def __init__(self, x, y, shuffle=True, batch_size = 32) :
        self.shuffle    = shuffle
        self.batch_size = batch_size
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        self.x = x.float()
        self.y = y.float()
        self.dataset = (self.x, self.y)
        
    
    def DataLoader(self) :
        self.training_data   = DataLoader(self.dataset,   batch_size=self.batch_size,shuffle=self.shuffle)
        return self.training_data


class MLP(nn.Module) :
    
    def __init__(self, model, epochs=2, learning_rate=0.01, momentum=0.9):
        super(MLP, self).__init__()
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.optimizer     = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_fn       = nn.CrossEntropyLoss()
        self.model         = model
   
    def train(self, data) :
        self.model.train()
        size = len(data.dataset)
        for batch, (X, y) in enumerate(data) :
            self.optimizer.zero_grad()
           
            y_hat = self.model(X) # prediction
            # CrossEntropyLoss does not accept one-hot coded tensor
            y= torch.argmax(y ,axis=1)
            self.loss = self.loss_fn(y_hat, y) # Loss
            self.loss.backward() # back prop
            self.optimizer.step() # update
            if batch % 100 == 0 :
                loss, current = self.loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
          
    def run(self, train_data) :
        
        for epoch in range(self.epochs) :
            print(f"epoch : {epoch+1}")
            self.train(train_data)
        
class SoftMax(nn.Module) :
    
    def __init__(self, input_size, hidden_size, output_size) :
        super(SoftMax,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fcl_1 = nn.Linear(self.input_size,  self.hidden_size)
        self.fcl_2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax_head = nn.Sequential(self.fcl_1, self.fcl_2, nn.Softmax(dim=1))
        
    
    def forward(self, x) :
        x = self.softmax_head(x)
        return x