import torch
import torch.nn as nn

class MathFormulaCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super(MathFormulaCRNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)) 
        )
        
        self.rnn = nn.GRU(
            input_size=128 * 4, 
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.conv(x) 
        b, c, h, w = features.size()

        features = features.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)  
        
        return logits.log_softmax(2)