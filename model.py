import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Attention Pooling and MaxPooling

class ADMIL(nn.Module):
    def __init__(self):
        super(ADMIL, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.attention_module = AttentionModule()
        self.classifier = Classifier()

    def forward(self, x, total_len):
        x = self.feature_extractor(x)       
        weights = self.attention_module(x)  # weights.shape: [N, 1]
        x = torch.sum(weights * x, dim=0, keepdim=True) 
        x = self.classifier(x)[0] # torch.Size([1])
        return x

class MIL_Max(nn.Module):
    def __init__(self):
        super(MIL_Max, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()

    def forward(self, x, total_len):
        x = self.feature_extractor(x)  
        x = torch.max(x, dim=0, keepdim=True)[0]  
        x = self.classifier(x)[0] 
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        # Flatten the bag and instance dimensions.
        x = x.view(-1, *x.shape[2:])
        x = self.conv(x)
        x = x.view(x.shape[0], -1)  # Flatten the output.
        return x


class AttentionModule(nn.Module):
    def __init__(self, d_model = 6272):
        super(AttentionModule, self).__init__()

        self.attention_network = nn.Sequential(
            nn.Linear(d_model, 512),  
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=0)  # Normalize weights to [0, 1] so they can be interpreted as probabilities
        )

    def forward(self, x):
        return self.attention_network(x)

class Classifier(nn.Module):
    def __init__(self, d_model = 6272):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

## SiSMIL

class SiSMIL(nn.Module):
    """
    One Stream-Trans Model
    """
    def __init__(self, feature_dim, num_heads, num_layers, ff_dim, output_dim, dropout=0.2, clip_ratio = 1):
        super().__init__()
        self.feature_extractor = VGG_Extractor()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock_Drop(d_model=feature_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.attention = AttentionModuleWithPosition(d_model =feature_dim)
        self.classifier = Classifier_Trans(d_model =feature_dim)
        self.clip_ratio = clip_ratio
        self.dropout = dropout

    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.feature_extractor(x)

        n = x.shape[0]
        if x.shape[0] == 1:
            x = x.unsqueeze(0)
            x_f = x
            for transformer_block in self.transformer_blocks:
                x_f = transformer_block(x_f, x_f, x_f)
            x_f = x_f.squeeze(0) 
            att_f = self.attention(x_f, total_len = n)
            weighted_sum_f = torch.matmul(att_f.transpose(0,1), x_f)
            weighted_sum_f = F.dropout(weighted_sum_f, self.dropout)
            x_f = self.classifier(weighted_sum_f)
            x_f = x_f[0,:]
        else:
            index = math.ceil(n * self.clip_ratio)
            outputs_f = []
            for i in range(index-1, n):
                x_f = x[0:i+1]
                x_f  = x_f.unsqueeze(0); 
                for transformer_block in self.transformer_blocks:
                    x_f = transformer_block(x_f, x_f, x_f)
                x_f = x_f.squeeze(0) 
                att_f = self.attention(x_f, total_len = n)
                weighted_sum_f = torch.matmul(att_f.transpose(0,1), x_f)
                weighted_sum_f = F.dropout(weighted_sum_f, self.dropout)
                output_f = self.classifier(weighted_sum_f)
                outputs_f.append(output_f[0,:])
            x_f = torch.stack(outputs_f)
        return x_f

class VGG_Extractor(nn.Module):
    def __init__(self):
        super(VGG_Extractor, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout(x)

        x = self.flatten(x)
        return x


class AttentionModuleWithPosition(nn.Module):
    def __init__(self):
        super(AttentionModuleWithPosition, self).__init__()
        self.linear1 = nn.Linear(288 + 2, 32)
        self.tanh1 = nn.Tanh()
        self.linear3 = nn.Linear(32 + 2, 1)
        self.softmax = nn.Softmax(dim=0)
        self.pos_embedding = PositionalEncoding(160)

    def forward(self, x, total_len):
        pos_encoding = self.pos_embedding(x, total_len)
        x = torch.cat([x, pos_encoding], dim=-1)
        x = self.tanh1(self.linear1(x))
        x = torch.cat([x, pos_encoding], dim=-1)
        x = self.softmax(self.linear3(x))
        return x

class Classifier_Trans(nn.Module):
    """
    A module for classifying transformed features.
    """
    def __init__(self, d_model = 3*3*64):
        super(Classifier_Trans, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

## BiSMIL

class BiSMIL(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, ff_dim, output_dim, dropout=0.2, clip_ratio=0.5):
        super().__init__()
        self.feature_extractor = VGG_Extractor()
        self.transformer_blocks_f = nn.ModuleList([
            TransformerBlock_Drop(d_model=feature_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)
        ]) 
        self.transformer_blocks_r = nn.ModuleList([
            TransformerBlock_Drop(d_model=feature_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.attention_f = AttentionModuleWithPosition()
        self.attention_r = AttentionModuleWithPosition()
        self.classifier_f = Classifier_Trans(d_model =feature_dim)
        self.classifier_r = Classifier_Trans(d_model =feature_dim)
        self.clip_ratio = clip_ratio
        self.dropout = dropout

    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.feature_extractor(x)
        n = x.shape[0]
        if x.shape[0] == 1:
            outputs_f = []
            outputs_r = []
            
            x = x.unsqueeze(0)
            x_f = x; x_r = x
            for transformer_block in self.transformer_blocks_f:
                x_f = transformer_block(x_f, x_f, x_f)
            x_f = x_f.squeeze(0) 
            att_f = self.attention_f(x_f, total_len = n)
            weighted_sum_f = torch.matmul(att_f.transpose(0,1), x_f)
            weighted_sum_f = F.dropout(weighted_sum_f, self.dropout)
            x_f = self.classifier_f(weighted_sum_f)
            x_f = x_f[0,:]
            
            for transformer_block in self.transformer_blocks_r:
                x_r = transformer_block(x_r, x_r, x_r)
            x_r = x_r.squeeze(0)
            att_r = self.attention_r(x_r, total_len = n)
            weighted_sum_r = torch.matmul(att_r.transpose(0,1), x_r)
            weighted_sum_r = F.dropout(weighted_sum_r, self.dropout)
            x_r = self.classifier_r(weighted_sum_r)
            x_r = x_r[0,:]
            
            outputs_f.append(x_f)
            outputs_r.append(x_r)
            
            x_f = outputs_f
            x_r = outputs_r
            #print("x_f", x_f)
        else:
            
            index = math.ceil(n * self.clip_ratio)
            outputs_f = []
            outputs_r = []
            for i in range(index-1, n):
                x_f = x[0:i+1]
                x_r = x[n-i-1:n].flip(dims=[0])  # Reverse the sequence
                x_f  = x_f.unsqueeze(0); x_r  = x_r.unsqueeze(0)
                for transformer_block in self.transformer_blocks_f:
                    x_f = transformer_block(x_f, x_f, x_f)
                x_f = x_f.squeeze(0) 
                att_f = self.attention_f(x_f, total_len = n)
                weighted_sum_f = torch.matmul(att_f.transpose(0,1), x_f)
                weighted_sum_f = F.dropout(weighted_sum_f, self.dropout)
                output_f = self.classifier_f(weighted_sum_f)
                outputs_f.append(output_f[0,:])
                
                for transformer_block in self.transformer_blocks_r:
                    x_r = transformer_block(x_r, x_r, x_r)
                x_r = x_r.squeeze(0)
                att_r = self.attention_r(x_r, total_len = n)
                weighted_sum_r = torch.matmul(att_r.transpose(0,1), x_r)
                weighted_sum_r = F.dropout(weighted_sum_r, self.dropout)
                output_r = self.classifier_r(weighted_sum_r)
                outputs_r.append(output_r[0,:])
            #print("outputs_r: ", outputs_r)
            x_f = torch.stack(outputs_f)
            x_r = torch.stack(outputs_r)
        #print("x_f", x_f)
        return x_f, x_r

## Transformer componet
class MultiHeadAttention(nn.Module):
    """
    A module implementing multi-head attention mechanism.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape and permute for multi-head attention
        Q = Q.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores and apply mask if provided
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()

        # Reshape and apply linear layer
        out = out.view(N, -1, self.d_model)
        out = self.linear(out)
        return out


class FeedForward(nn.Module):
    """
    A feed-forward neural network module used in Transformer layers.
    """
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class TransformerBlock_Drop(nn.Module):
    """
    A Transformer block module with dropout for regularization.
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(query, key, value, mask)
        x = self.norm1(attention + query)
        x = self.dropout(x)
        ff = self.ff(x)
        x = self.norm2(ff + x)
        x = self.dropout(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, scaling_factor=5):
        super(PositionalEncoding, self).__init__()
        self.position = nn.Parameter(torch.arange(max_len).float(
        ), requires_grad=False)  # 1D Parameter with size of max_len
        self.scaling_factor = scaling_factor

    def forward(self, x, total_len):
        # Linear position encoding
        if total_len == 1:
            linear_position = torch.tensor([0.5]).float()
        else:
            linear_position = self.position[:x.shape[0]] / (total_len - 1)  # Normalize relative position to range [0, 1]
        # shape: [x.shape[0], 1]
        linear_pos_encoding = linear_position.unsqueeze(
            -1) * self.scaling_factor
        # Gaussian position encoding
        gaussian_position = torch.exp(-((self.position[:x.shape[0]] - total_len / 2) ** 2) / (
            2 * (total_len / 2) ** 2))  # Gaussian distribution
        # shape: [x.shape[0], 1]
        gaussian_pos_encoding = gaussian_position.unsqueeze(-1) * self.scaling_factor
        # Concatenation
        # shape: [x.shape[0], 2]
        linear_pos_encoding = linear_pos_encoding.to(device)
        gaussian_pos_encoding = gaussian_pos_encoding.to(device)
        pos_encoding = torch.cat([linear_pos_encoding, gaussian_pos_encoding], dim=-1)

        return pos_encoding
    


class MILAttentionLayer(nn.Module):
    def __init__(self, weight_params_dim, use_gated=False):
        super(MILAttentionLayer, self).__init__()
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.v = nn.Linear(weight_params_dim, weight_params_dim)
        self.w = nn.Linear(weight_params_dim, 1)
        if self.use_gated:
            self.u = nn.Linear(weight_params_dim, weight_params_dim)

    def forward(self, x):
        original_x = x
        x = torch.tanh(self.v(x))
        if self.use_gated:
            # x *= torch.sigmoid(self.u(original_x))
            x = x * torch.sigmoid(self.u(original_x))
        attention_scores = self.w(x)
        alpha = F.softmax(attention_scores, dim=0)
        return alpha


## SA_DMIL Model
class SmoothMIL(nn.Module):
    def __init__(self, alpha=0.5, S_k=1):
        super(SmoothMIL, self).__init__()
        self.alpha = alpha
        self.S_k = S_k

    def compute_Laplacian(self, bag_size):
        G = nx.Graph()
        for e in range(bag_size - 1):
            G.add_edge(e + 1, e + 2)
        degree_matrix = np.diag(list(dict(G.degree()).values())) + np.eye(bag_size)
        adjacency_matrix = nx.adjacency_matrix(G).toarray()
        L = torch.tensor(degree_matrix - adjacency_matrix, dtype=torch.float32)
        return L

    def forward(self, y_pred, y_true, att_weights):
        bag_size = att_weights.shape[0] #
        if bag_size == 1:
            loss_combined = F.binary_cross_entropy(y_pred, y_true)
            return loss_combined
        L = self.compute_Laplacian(bag_size)

        att_weights = att_weights.to(device)
        L = L.to(device)


        loss1 = F.binary_cross_entropy(y_pred, y_true)
        att_weights = torch.transpose(att_weights, 1, 0)
        
        if self.S_k == 1:
            VV = torch.matmul(att_weights, L)
            loss2 = torch.matmul(VV, torch.transpose(att_weights, 1, 0))
        elif self.S_k == 2:
            VV = torch.matmul(att_weights, L)
            VV = torch.matmul(VV, L)
            loss2 = torch.matmul(VV, torch.transpose(att_weights, 1, 0))

        loss2 = torch.mean(loss2)
        loss_combined = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss_combined


class SA_DMIL(nn.Module):
    def __init__(self):
        super(SA_DMIL, self).__init__()
        self.vgg = VGG_Extractor()
        self.attention = MILAttentionLayer(weight_params_dim=288, use_gated=True)
        self.fc1 = nn.Linear(288, 128)  # Updated the input dimension to 288
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        h = self.vgg(x)
        a = self.attention(h)
        h = h.view(-1, h.shape[0], h.shape[1])
        weighted_sum = torch.matmul(a.transpose(0,1), h.squeeze(0))  
        intermediate = F.dropout(weighted_sum, 0.25) 
        intermediate = F.relu(self.fc1(intermediate))
        out = torch.sigmoid(self.fc2(intermediate))
        return out[0], a 

## utility function
def construct_sequence(n, temperature=2.0):
    """
    Constructs a sequence from 1 to n, then applies the softmax function to the sequence.
    """
    # Construct a sequence from 1 to n.
    sequence = np.arange(1, n + 1)

    # Apply the softmax function to the sequence.
    e_x = np.exp((sequence - np.max(sequence)) / temperature)  # Subtract the max for numerical stability
    softmax_sequence = e_x / e_x.sum()

    return softmax_sequence

def compute_weighted_incremental_loss(outputs, labels, softmax_sequence):

    loss = nn.BCELoss()
    losses = []
    
    for i, output in enumerate(outputs):
        if len(output.shape) == 0:
            output = output.unsqueeze(0)
        instance_loss = loss(output, labels)
        weighted_instance_loss = instance_loss * softmax_sequence[i]
        losses.append(weighted_instance_loss)
        
    return sum(losses)