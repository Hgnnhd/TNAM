import torch
import torch.nn as nn
import torch.nn.functional as F

class TCell(nn.Module):
    def __init__(self, input_size, hidden_size, k):
        super(TCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.k = k # K is the number of variables in the input
        # Parameters for input, forget, and output gates
        self.w_i = nn.Parameter(torch.empty(hidden_size*self.k, input_size))
        self.w_f = nn.Parameter(torch.empty(hidden_size*self.k, input_size))
        self.w_o = nn.Parameter(torch.empty(hidden_size*self.k, input_size))
        self.w_c = nn.Parameter(torch.empty(hidden_size*self.k, input_size))

        self.r_i = nn.Parameter(
            torch.empty(hidden_size*self.k, hidden_size)
        )
        self.r_f = nn.Parameter(
            torch.empty(hidden_size*self.k, hidden_size)
        )
        self.r_o = nn.Parameter(
            torch.empty(hidden_size*self.k, hidden_size)
        )
        self.r_c = nn.Parameter(
            torch.empty(hidden_size*self.k, hidden_size)
        )

        self.b_i = nn.Parameter(torch.empty(self.k,hidden_size*self.k))
        self.b_f = nn.Parameter(torch.empty(self.k,hidden_size*self.k))
        self.b_o = nn.Parameter(torch.empty(self.k,hidden_size*self.k))
        self.b_c = nn.Parameter(torch.empty(self.k,hidden_size*self.k))


        self.sigmoid = nn.Sigmoid()
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_o)
        nn.init.xavier_uniform_(self.w_c)

        nn.init.xavier_uniform_(self.r_i)
        nn.init.xavier_uniform_(self.r_f)
        nn.init.xavier_uniform_(self.r_o)
        nn.init.xavier_uniform_(self.r_c)

        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_c)

    def decompose_optimized(self, x):
        batch_size, rows, cols = x.size()
        k = cols // rows
        # Extract submatrices using gather
        row_indices = torch.arange(rows, device=x.device).unsqueeze(0)
        col_indices = (row_indices * k).unsqueeze(2) + torch.arange(k, device=x.device).unsqueeze(0).unsqueeze(1)
        col_indices = col_indices.expand(batch_size, rows, k)
        new_x = x.gather(2, col_indices)
        return new_x

    def forward(self, x, states):

        h_prev, c_prev = states

        batch = x.size(0)

        w_o = (self.w_o)
        w_c = (self.w_c)
        r_o = (self.r_o)
        r_c = (self.r_c)
        b_o = (self.b_o)
        b_c = (self.b_c)

        i_tilda = torch.matmul(x, self.w_i.t()) + torch.matmul(h_prev, self.r_i.t()) + self.b_i.unsqueeze(0).repeat(batch,1,1)
        f_tilda = torch.matmul(x, self.w_f.t()) + torch.matmul(h_prev, self.r_f.t()) + self.b_f.unsqueeze(0).repeat(batch,1,1)
        o_tilda = torch.matmul(x, w_o.t()) + torch.matmul(h_prev, r_o.t()) + b_o.unsqueeze(0).repeat(batch,1,1)
        c_tilda = torch.matmul(x, w_c.t()) + torch.matmul(h_prev, r_c.t()) + b_c.unsqueeze(0).repeat(batch,1,1)

        if self.k>1:
            i_tilda = self.decompose_optimized(i_tilda)
            f_tilda = self.decompose_optimized(f_tilda)
            o_tilda = self.decompose_optimized(o_tilda)
            c_tilda = self.decompose_optimized(c_tilda)


        # # 计算门控值
        i = torch.sigmoid(i_tilda)
        f = torch.sigmoid(f_tilda)
        o = torch.sigmoid(o_tilda)

        # # 更新细胞状态和隐藏状态
        c_t = f * c_prev + i * c_tilda
        h_t = o * (torch.tanh(c_t))

        return h_t, c_t

class TCell_new(nn.Module):
    def __init__(self, input_size, hidden_size, k):
        super(TCell_new, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k  # number of variables

        self.w = nn.Parameter(torch.empty(self.k, input_size + hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.empty(self.k, 4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.k):
            nn.init.xavier_uniform_(self.w[i])
        nn.init.zeros_(self.b)

    def forward(self, x, states):
        h_prev, c_prev = states

        combined = torch.cat([x, h_prev], dim=-1)
        gates = torch.einsum("bni,nio->bno", combined, self.w) + self.b.unsqueeze(0)
        i_tilda, f_tilda, o_tilda, c_tilda = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_tilda)
        f = torch.sigmoid(f_tilda)
        o = torch.sigmoid(o_tilda)

        c_t = f * c_prev + i * c_tilda
        h_t = o * torch.tanh(c_t)

        return h_t, c_t

class TNAM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0, time=6, dataset_name="TJ"):
        super(TNAM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time = time
        self.layers = nn.ModuleList([TCell_new(input_size=1, hidden_size=hidden_size, k=input_size)
                                     for i in range(num_layers)])
        self.fc_var = nn.Linear(input_size, 1)
        self.fc_t = nn.Linear(time, 1)
        self.fc = nn.Linear(hidden_size * 1, 1)
        self.fc_emb = nn.Linear(1, hidden_size)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.in_layernorm = nn.LayerNorm(input_size)
        self.norm_flag = True

        self.attention_net = nn.Sequential(
            nn.Linear(self.input_size * self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

    def time_attention(self, hidden_states):
        """基于LSTM隐状态序列的时间注意力"""
        # hidden_states: [B, T, N, D]
        B, T, N, D = hidden_states.size()
        hidden_states = hidden_states.view(B, T, N * D)  # [B, T, N*D]
        energy = self.attention_net(hidden_states)  # [B, T, 1]
        attention_weights = F.softmax(energy, dim=1)
        return attention_weights

    def C_relu(self, z):
        # 分别对实部和虚部应用ReLU
        activated_real = torch.relu(z.real)
        activated_imag = torch.relu(z.imag)
        # 合并实部和虚部以形成复数
        return torch.complex(activated_real, activated_imag)

    def C_elu(self, z):
        activated_real = F.elu(z.real)
        activated_imag = F.elu(z.imag)
        # 合并实部和虚部以形成复数
        return torch.complex(activated_real, activated_imag)

    def fc_att(self,x):

        if x.dim() == 3:
            squeeze = F.adaptive_avg_pool1d(x, 1)
        else:
            squeeze = F.adaptive_avg_pool2d(x, (1, 1))

        if squeeze.is_complex():
            # 使用自定义的复数ReLU激活函数
            excitation = self.C_relu(squeeze)
        else:
            # 使用标准的ReLU激活函数
            excitation = F.relu(squeeze)

        scale = torch.sigmoid(excitation)
        scale = scale.expand_as(x)
        return x * scale

    def inter_op(self, matrix):

        matrix = torch.fft.rfft(matrix, dim=1, norm='ortho')  # FFT on N dimension
        matrix1 = matrix
        matrix2 = matrix
        x_f = matrix1[:,:,None,:] * matrix2[:,None,:,:]

        x_f = self.fc_att(x_f)
        x_f = torch.sum(x_f,dim=1)

        multiplied_features = torch.fft.irfft(x_f, dim=1, norm='ortho')  # FFT on N dimension
        multiplied_features = self.fc_att(multiplied_features)
        multiplied_features = torch.sum(multiplied_features, dim=1)

        return multiplied_features

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        initial_states = [torch.zeros(batch_size, self.input_size, self.layers[0].hidden_size).to(x.device)] * 2
        outputs = []
        current_states = initial_states
        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(-1)
            new_states = []
            for i,layer in enumerate(self.layers):
                h_t, new_state = layer(x_t, current_states)
                new_states.append(new_state)
                x_t = h_t  # Pass the output to the next layer
            outputs.append(h_t)
            current_states = (h_t,new_state)

        lstm_out_v = torch.stack(outputs, dim=-1) #[B, N, D, T]
        lstm_out_v = (lstm_out_v.permute(0, 3, 1, 2))  # [B, T, N, D]
        # 使用时间注意力进行加权聚合，替代简单的sum
        # 基于LSTM输出序列计算注意力权重
        # time_scores = self.time_attention(lstm_out_v)  # [B, T, 1]
        # weighted_features = lstm_out_v * time_scores.unsqueeze(-1)
        # lstm_out_v = weighted_features.sum(dim=1)  # [B, N, D]
        # lstm_out_v = self.fc_att(lstm_out_v)  # [B, T, N, D]
        # Keep the batch dimension even when B==1 (squeeze() would break inter_op).
        lstm_out_v = torch.sum(lstm_out_v, dim=1)  # [B, N, D]
        # lstm_out_v = self.layernorm(lstm_out_v)
        lstm_out_v = self.inter_op(lstm_out_v) # [B, D]
        lstm_out_v = self.layernorm(lstm_out_v)
        lstm_out_v = self.dropout(lstm_out_v)
        lstm_out_v = self.fc(lstm_out_v)  # [B, 1]
        lstm_out = self.sigmoid(lstm_out_v)

        return lstm_out
