import torch
import torch.nn as nn

class TCell(nn.Module):
    def __init__(self, input_size, hidden_size, k):
        super(TCell, self).__init__()
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


        i = torch.sigmoid(i_tilda)
        f = torch.sigmoid(f_tilda)
        o = torch.sigmoid(o_tilda)

        c_t = f * c_prev + i * c_tilda
        h_t = o * (torch.tanh(c_t))

        return h_t, c_t

class TNAM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0):
        super(TNAM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([TCell(input_size=1, hidden_size=hidden_size, k=input_size)
                                     for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(hidden_size)



    def inter_op(self, matrix):

        matrix = torch.fft.rfft(matrix, dim=1, norm='ortho')  # FFT on N dimension
        matrix1 = matrix
        matrix2 = matrix
        x_f = matrix1[:,:,None,:] * matrix2[:,None,:,:]
        x_f = torch.sum(x_f,dim=1)
        multiplied_features = torch.fft.irfft(x_f, dim=1, norm='ortho')  # FFT on N dimension
        multiplied_features = torch.sum(multiplied_features, dim=1)

        return multiplied_features

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        initial_states = [torch.zeros(batch_size, self.input_size, self.hidden_size).to(x.device)] * 2
        outputs = []
        current_states = initial_states
        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(-1)
            for layer in self.layers:
                h_t, new_state = layer(x_t, current_states)
                x_t = h_t
            outputs.append(h_t)
            current_states = (h_t,new_state)

        lstm_out_v = torch.stack(outputs, dim=-1) #[B, N, D, T]
        lstm_out_v = (lstm_out_v.permute(0, 3, 1, 2))  # [B, T, N, D]
        lstm_out_v = torch.sum(lstm_out_v, dim=1)  # [B, N, D]
        lstm_out_v = self.layernorm(lstm_out_v)
        lstm_out_v = self.inter_op(lstm_out_v) # [B, D]
        lstm_out_v = self.layernorm(lstm_out_v)
        lstm_out_v = self.dropout(lstm_out_v)
        lstm_out_v = self.fc(lstm_out_v)  # [B, 1]
        lstm_out = self.sigmoid(lstm_out_v)

        return lstm_out
