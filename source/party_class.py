from source.model import CNN_bottom, ResNet18_bottom, top
import torch.optim as optim
class Passive_Party:
    def __init__(self, model, in_channel, img_size, embed_dim):
        if model == 'cnn':
            self.model = CNN_bottom(in_channel, img_size, embed_dim)
        elif model == 'resnet':
            self.model = ResNet18_bottom(in_channel, img_size, embed_dim)
    def set_optimizer(self, lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr)
    def update_model(self):
        self.optimizer.step()


class Active_Party:
    def __init__(self, embed_dim, class_num):
        self.model = top(embed_dim, class_num)
    def set_optimizer(self, lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr)
    def update_model(self):
        self.optimizer.step()
