
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class DEASRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(DEASRec, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.max_seq_len = config["MAX_ITEM_LIST_LENGTH"]

        self.lambda_kl = config["lambda_kl"]
        self.kl_annealing_steps = config['kl_annealing_steps']
        self.kl_annealing = config['kl_annealing']
        self.l2_reg_weight = config['l2_reg_weight']

        self.max_noise_scale = config['max_noise_scale']

        self.num_subspaces = config["num_subspaces"]
        self.epsilon = config["epsilon"]
        self.pvm_use_layer_norm = config['pvm_use_layer_norm']

        # Ablation switches
        self.use_pvm = config['use_pvm']
        self.use_asr = config['use_asr']

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.use_pvm:
            self.PVM = PolarVectorModulator(
                self.hidden_size,
                self.max_seq_len,
                num_subspaces=self.num_subspaces,
                epsilon=self.epsilon,
                use_layer_norm=self.pvm_use_layer_norm
            )

        # Conditionally create ASR layers (only when enabled)
        if self.use_asr:
            self.noise_activation = AdaptiveSmoothActivation(initial_slope=0.9)
            self.mean_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.log_var_layer = nn.Linear(self.hidden_size, self.hidden_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)

        # item_emb = self.filter_layers(item_emb)
        # PVM: Polar Vector Modulation (controllable via config)
        if self.use_pvm:
            item_emb = self.PVM(item_emb)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        last_hidden_state = self.gather_indexes(output, item_seq_len - 1)

        # When ASR is disabled, return directly like original SASRec
        if not self.use_asr:
            return last_hidden_state, None, None

        # ASR enabled: compute mean and variance
        mean = self.mean_layer(last_hidden_state)
        log_var = self.log_var_layer(last_hidden_state)

        if self.training:
            # Stochastic smoothing during training
            std = self.max_noise_scale * self.noise_activation(log_var) + 1e-6
            eps = torch.randn_like(std)
            seq_output = last_hidden_state + mean + eps * std
        else:
            # Deterministic output during inference
            seq_output = last_hidden_state + mean
            std = torch.zeros_like(mean)
        return seq_output, mean, std

    def calculate_loss(self, interaction):

        if self.training:
            self.global_step += 1

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # seq_output is the sampled z from the posterior q(z|x)
        seq_output, mean, std = self.forward(item_seq, item_seq_len)

        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)

        # When ASR is disabled, behave exactly like original SASRec
        if not self.use_asr:
            return rec_loss

        # ASR enabled: add KL loss and L2 regularization
        kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(std) - mean.pow(2) - std.pow(2), dim=-1)
        kl_loss = torch.mean(kl_loss)

        if self.training and self.kl_annealing:
            current_lambda = self.lambda_kl * min(1.0, self.global_step.float() / self.kl_annealing_steps)
        else:
            current_lambda = self.lambda_kl

        l2_reg_weight = self.l2_reg_weight
        if l2_reg_weight > 0:
            l2_reg_loss = self.item_embedding.weight.norm(p=2).pow(2) / 2
        else:
            l2_reg_loss = 0.0

        loss = rec_loss + current_lambda * kl_loss + l2_reg_weight * l2_reg_loss

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, mean, std = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, mean, std = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class AdaptiveSmoothActivation(nn.Module):

    def __init__(self, initial_slope=1.0, eps=1e-4):
        super(AdaptiveSmoothActivation, self).__init__()
        self.k_param = nn.Parameter(torch.tensor(1.0 / initial_slope))
        self.eps = eps

    def forward(self, x):
        k = F.softplus(self.k_param) + 1e-6
        smooth_abs = torch.sqrt(x.pow(2) + self.eps)

        # y = x / (k + |x|)
        y = x / (k + smooth_abs)

        # map to (0, 1)
        return 0.5 * (1.0 + y)


class PolarVectorModulator(nn.Module):
    def __init__(self, embed_dim, seq_len, num_subspaces=8, epsilon=0.3, use_layer_norm=False):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.epsilon = epsilon
        self.use_layer_norm = use_layer_norm

        trans_dim = seq_len // 2 + 1

        self.subspace_stride = trans_dim // num_subspaces

        self.radial_gain = nn.Parameter(torch.zeros(num_subspaces, 1, embed_dim))
        self.angular_shift = nn.Parameter(torch.zeros(num_subspaces, 1, embed_dim))

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x):

        if self.use_layer_norm:
            x = self.layer_norm(x)

        B, N, D = x.shape

        z_trans = torch.fft.rfft(x, dim=1)
        radius = torch.abs(z_trans)
        theta = torch.angle(z_trans)
        z_modulated = torch.zeros_like(z_trans)

        current_dim = z_trans.shape[1]

        for i in range(self.num_subspaces):
            idx_start = i * self.subspace_stride

            if idx_start >= current_dim:
                break

            is_last = (i == self.num_subspaces - 1)
            idx_end = current_dim if is_last else (i + 1) * self.subspace_stride

            if idx_end > current_dim:
                idx_end = current_dim

            gain = torch.tanh(self.radial_gain[i].unsqueeze(0)) * self.epsilon
            shift = torch.tanh(self.angular_shift[i].unsqueeze(0)) * self.epsilon

            radius_part = radius[:, idx_start:idx_end, :] * (1 + gain)
            theta_part = theta[:, idx_start:idx_end, :] + shift
            z_modulated[:, idx_start:idx_end, :] = torch.polar(radius_part, theta_part)

        x_out = torch.fft.irfft(z_modulated, n=N, dim=1)

        return x_out

