import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

class SparseDataset3Input(Dataset):
    def __init__(self, y_int, y_clean, signals, angles, svs, num_samples=None):
        super().__init__()
        self.y_int = y_int
        self.y_clean = y_clean
        self.signals = signals
        self.angles = angles
        self.svs = svs
        self.num_samples = num_samples if num_samples is not None else len(y_int)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.y_int[idx], self.y_clean[idx], self.signals[idx], self.angles[idx], self.svs[idx]

class LISTA_Net(nn.Module):
    def __init__(self, kernel_strategy="decreasing", dropout_rate=0.1):
        super(LISTA_Net, self).__init__()
        self.num_layers = 1
        self.kernel_strategy = kernel_strategy
        self.dropout_rate = dropout_rate

        self.conv_y_layers = nn.ModuleList()
        self.conv_angle_layers = nn.ModuleList()

        channels = [512, 256, 128, 128, 128, 
                    80, 64, 64, 64, 64, 64, 
                    48, 32, 32, 32, 32, 32]

        kernel_sizes = [
            #layer 1
            (1, 3), #1x3
            (1, 3), #1x5
            (1, 3), #1x7
            (1, 3), #1x9
            (1, 3), #1x11
            #layer 6
            (3, 3), #3x13
            (1, 3), #3x15
            (1, 3), #3x17
            (1, 3), #3x19
            (1, 3), #3x21
            (1, 3), #3x23
            #layer 12
            (3, 3), #4x25
            (1, 3), #4x27
            (1, 3), #4x29
            (1, 3), #4x31
            (1, 3), #4x33
        ]
        
        paddings = [(kh // 2, kw // 2) for kh, kw in kernel_sizes]

        for _ in range(self.num_layers):
            layers = []
            for i, (in_ch, out_ch, k, pad) in zip(
                range(len(kernel_sizes)), 
                zip(channels[:-1], channels[1:], kernel_sizes, paddings)
            ):
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad))

                if out_ch != 2:
                    layers.append(nn.LayerNorm(normalized_shape=(out_ch, 4, 32)))
                    layers.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*layers)
            self.conv_y_layers.append(block)


        angle_layers = []
        angle_layers.append(nn.Conv2d(channels[-1]+2, 16, (3, 3),padding=(1,1) ))
        angle_layers.append(nn.Conv2d(16, 8, (3, 3),padding=(1,1)  ))
        angle_layers.append(nn.Conv2d(8, 4, (1, 3), padding=(0,1) ))
        angle_layers.append(nn.Conv2d(4, 2, (1, 3), padding=(0,1) ))                
        self.conv_angle_layers.append(nn.Sequential(*angle_layers))

    def clean_y(self, y):
        out = y
        for i in range(self.num_layers):
            out = self.conv_y_layers[i](out)            
        return out

    def angled_y(self, y, phi, theta):

        out = torch.concat([y, phi, theta], dim=1)
        out = self.conv_angle_layers[0](out)        
        return out
    
    def forward(self, y_input, phi, theta):
        print(y_input.shape)
        out = self.clean_y(y_input);
        print(out.shape)
        out = self.angled_y(out, phi, theta)
        
        return out

    # def forward(self, y_input, angles_batch):
    #     out = self.clean_y(y_input)

    #     out_angles = torch.zeros((y_input.shape[0],angles_batch.shape[1], 2, 4, 32), dtype=torch.float32).to(y_input.device)
    #     for i in range(angles_batch.shape[1]):
    #         theta = angles_batch[:, i, 0].view(y_input.shape[0], 1, 1, 1).expand(-1, 1, 4, 32)
    #         phi = angles_batch[:, i, 1].view(y_input.shape[0], 1, 1, 1).expand(-1, 1, 4, 32)
    #         out_angles[:,i] = self.angled_y(out, phi, theta)        
    #     return out_angles