import torch
import math
from dataclasses import dataclass

from env import Actions, GenerateConfig, OutputMetadata, StateFeatures

NUM_COORD_VALUES = 256
COORD_OFFSET = 128

# These tensors are all f32 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class Predictions:
    # log-odds of invalid door (unconnected):
    door_invalid: torch.Tensor
    # log-odds of invalid connection (lack of return path):
    connection_invalid: torch.Tensor


def get_predictions(raw_preds, output_sizes):
    preds = []
    col = 0
    for size in output_sizes:
        preds.append(raw_preds[:, :, col:(col + size)])
        col += size

    return Predictions(
        door_invalid=preds[0],
        connection_invalid=preds[1],
    )


def normalize(x: torch.Tensor):
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


class FactorizedOutcomeHead(torch.nn.Module):
    def __init__(self, output_metadata, num_geometry_outcomes, embedding_width):
        super().__init__()
        self.embedding_width = embedding_width
        self.num_outputs = len(output_metadata)
        metadata = torch.tensor(output_metadata, dtype=torch.int64).reshape(self.num_outputs, 2)
        self.register_buffer("room_idx", metadata[:, 0])
        self.register_buffer("geometry_outcome_idx", metadata[:, 1])
        self.geometry_outcome_embedding = torch.nn.Parameter(
            torch.randn([num_geometry_outcomes, embedding_width]) / math.sqrt(embedding_width))
        self.state = torch.nn.Linear(embedding_width, embedding_width, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(math.sqrt(embedding_width) / 2)))

    def forward(self, X, room_x, room_y, room_placed, pos_embedding_x, pos_embedding_y):
        if self.num_outputs == 0:
            return X.new_empty([X.shape[0], X.shape[1], 0], dtype=torch.float32)
        state = self.state(X)
        # Keep normalization, base logits, and final logits out of reduced
        # precision. These scores directly drive both the loss and candidate
        # selection.
        with torch.amp.autocast(X.device.type, enabled=False):
            state = torch.nn.functional.normalize(state.to(torch.float32), dim=-1)
            geometry_outcome_embedding = torch.nn.functional.normalize(
                self.geometry_outcome_embedding.to(torch.float32), dim=-1)
            pos_embedding_x = torch.nn.functional.normalize(pos_embedding_x.to(torch.float32), dim=-1)
            pos_embedding_y = torch.nn.functional.normalize(pos_embedding_y.to(torch.float32), dim=-1)
            base_query = geometry_outcome_embedding[self.geometry_outcome_idx]
            base_logits = torch.matmul(state, base_query.transpose(0, 1))
        x_logits = torch.matmul(state, pos_embedding_x.transpose(0, 1))
        y_logits = torch.matmul(state, pos_embedding_y.transpose(0, 1))
        room_logits = torch.gather(x_logits, -1, room_x) + torch.gather(y_logits, -1, room_y)
        room_logits = torch.where(room_placed, room_logits, 0.0)
        position_logits = room_logits[..., self.room_idx]
        return (base_logits + position_logits) * torch.exp(
            torch.clamp(self.logit_scale, max=math.log(100.0))
        )


class GroupedQueryAttentionLayer(torch.nn.Module):
    def __init__(self, input_width, key_width, value_width, num_heads, num_groups):
        super().__init__()
        self.input_width = input_width
        self.key_width = key_width
        self.value_width = value_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        assert num_heads % num_groups == 0
        self.num_heads_per_group = num_heads // num_groups
        self.query = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.key = torch.nn.Linear(input_width, num_groups * key_width, bias=False)
        self.value = torch.nn.Linear(input_width, num_groups * value_width, bias=False)
        self.post = torch.nn.Linear(num_heads * value_width, input_width, bias=False)
        # self.post.weight.data.zero_()
        # self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == self.input_width
        n = X.shape[0]  # batch dimension
        s = X.shape[1]  # sequence dimension
        Q = self.query(X).view(n, s, self.num_heads, self.key_width).transpose(1, 2)
        Q = normalize(Q)
        K = self.key(X).view(n, s, self.num_groups, self.key_width).transpose(1, 2)
        K = normalize(K)
        V = self.value(X).view(n, s, self.num_groups, self.value_width).transpose(1, 2)
        # A = compute_grouped_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)

        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=X.device))
        causal_mask = causal_mask & ~torch.eye(s, dtype=torch.bool, device=X.device)

        A = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True, attn_mask=causal_mask)
        # A: [n, h, s, v]
        A = A.transpose(1, 2).reshape(n, s, self.num_heads * self.value_width)
 
        P = self.post(A)
        # print("forward: Q:", Q.shape, Q, "\nK:", K.shape, K, "\nV:", V.shape, V, "\nA:", A.shape, A, "\nP:", P.shape, P)
        # out = self.layer_norm(X + P).to(X.dtype)
        # P = self.layer_norm(P).to(X.dtype)
        return X + P


class FeedforwardLayer(torch.nn.Module):
    def __init__(self, input_width, hidden_width):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_width, hidden_width, bias=False)
        self.lin2 = torch.nn.Linear(hidden_width, input_width, bias=False)

    def forward(self, X):
        A = normalize(X)
        A = self.lin1(A)
        A = torch.nn.functional.gelu(A)
        A = self.lin2(A)
        return X + A


class CausalTransformerModel(torch.nn.Module):
    def __init__(self, num_rooms, output_metadata: OutputMetadata, map_x, map_y, embedding_width, key_width, value_width, attn_heads, head_groups, hidden_width, num_layers):
        super().__init__()
        self.num_rooms = num_rooms
        self.map_x = map_x
        self.map_y = map_y
        self.num_tokens = self.num_rooms + 1
        self.output_sizes = output_metadata.get_output_sizes()
        self.num_outputs = sum(self.output_sizes)
        self.num_layers = num_layers
        self.embedding_width = embedding_width
        self.global_lin = torch.nn.Linear(1, embedding_width)
        self.pos_embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
        room_connection_variant_idx = torch.tensor(
            output_metadata.room_connection_variant_idx + [output_metadata.num_room_connection_variants],
            dtype=torch.int64)
        assert room_connection_variant_idx.shape == (self.num_rooms + 1,)
        self.register_buffer("room_connection_variant_idx", room_connection_variant_idx)
        self.connection_variant_embedding = torch.nn.Parameter(
            torch.randn([output_metadata.num_room_connection_variants + 1, embedding_width])
            / math.sqrt(embedding_width))
        self.room_embedding = torch.nn.Parameter(
            torch.randn([num_rooms + 1, embedding_width])
            / (10.0 * math.sqrt(embedding_width)))
        self.attn_layers = torch.nn.ModuleList()
        self.ff_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            attn_layer = GroupedQueryAttentionLayer(
                input_width=embedding_width,
                key_width=key_width,
                value_width=value_width,
                num_heads=attn_heads,
                num_groups=head_groups)
            self.attn_layers.append(attn_layer)
            ff_layer = FeedforwardLayer(
                input_width=embedding_width,
                hidden_width=hidden_width)
            self.ff_layers.append(ff_layer)

        self.door_output = FactorizedOutcomeHead(
            output_metadata.door, output_metadata.num_door_variants, embedding_width)
        self.connection_output = FactorizedOutcomeHead(
            output_metadata.connection, output_metadata.num_connection_variants, embedding_width)


    def get_embedding(self, room_idx, room_x, room_y, config: GenerateConfig):
        # global_data = torch.cat([torch.log(config.temperature.view(-1, 1))], dim=1)

        # global_emb = self.global_lin(global_data).unsqueeze(1)
        # TODO: try rotary positional embeddings
        position_emb_x = self.pos_embedding_x[room_x + COORD_OFFSET]
        position_emb_y = self.pos_embedding_y[room_y + COORD_OFFSET]
        conn_var_emb = self.connection_variant_embedding[self.room_connection_variant_idx[room_idx]]
        room_emb = self.room_embedding[room_idx]
        
        # X = global_emb + position_emb_x + position_emb_y + room_emb
        X = position_emb_x + position_emb_y + room_emb + conn_var_emb
        return X        

    def get_placement_state(self, room_idx, room_x, room_y):
        valid_room = room_idx < self.num_rooms
        room_one_hot = torch.nn.functional.one_hot(
            torch.where(valid_room, room_idx, 0), self.num_rooms)
        room_one_hot = room_one_hot & valid_room.unsqueeze(-1)
        room_placed = torch.cumsum(room_one_hot, dim=1) > 0
        placed_x = torch.cumsum(room_one_hot * room_x.unsqueeze(-1), dim=1)
        placed_y = torch.cumsum(room_one_hot * room_y.unsqueeze(-1), dim=1)
        return placed_x, placed_y, room_placed

    def get_output(self, X, room_x, room_y, room_placed):
        door = self.door_output(
            X, room_x + COORD_OFFSET, room_y + COORD_OFFSET, room_placed,
            self.pos_embedding_x, self.pos_embedding_y)
        connection = self.connection_output(
            X, room_x + COORD_OFFSET, room_y + COORD_OFFSET, room_placed,
            self.pos_embedding_x, self.pos_embedding_y)
        return torch.cat([door, connection], dim=-1)


    def forward(self, actions: Actions, config: GenerateConfig):
        room_idx = actions.room_idx.to(torch.int64)
        room_x = actions.room_x.to(torch.int64)
        room_y = actions.room_y.to(torch.int64)

        with torch.amp.autocast(
            'cuda',
            dtype=torch.bfloat16,
            enabled=room_idx.device.type == 'cuda' and config.training_autocast,
        ):
            X = self.get_embedding(room_idx, room_x, room_y, config)
            # print("forward: X:", X.shape, X)
            for i in range(len(self.attn_layers)):
                X = self.attn_layers[i](X)
                X = self.ff_layers[i](X)

        X = normalize(X)
        placed_x, placed_y, room_placed = self.get_placement_state(room_idx, room_x, room_y)
        X = self.get_output(X, placed_x, placed_y, room_placed)
        X = X.to(torch.float32)
        return get_predictions(X, self.output_sizes)


    def get_initial_kv_cache(self, batch_size, device):
        K_list = []
        V_list = []
        for layer in self.attn_layers:
            g = layer.num_groups
            K_list.append(torch.zeros([batch_size, g, 0, layer.key_width], device=device))
            V_list.append(torch.zeros([batch_size, g, 0, layer.value_width], device=device))
        room_x = torch.zeros([batch_size, self.num_rooms], dtype=torch.int64, device=device)
        room_y = torch.zeros([batch_size, self.num_rooms], dtype=torch.int64, device=device)
        room_placed = torch.zeros([batch_size, self.num_rooms], dtype=torch.bool, device=device)
        return K_list, V_list, room_x, room_y, room_placed


    def get_updated_kv_cache(self, old_kv_cache, cache_candidates, action_idx):
        old_K_list, old_V_list, _, _, _ = old_kv_cache
        cand_K_list, cand_V_list, cand_room_x, cand_room_y, cand_room_placed = cache_candidates
        new_K_list = []
        new_V_list = []
        for old_K, old_V, cand_K, cand_V in zip(old_K_list, old_V_list, cand_K_list, cand_V_list):
            # old_K: [b, g, s, k]
            # cand_K: [b, g, c, k]
            batch_idx = torch.arange(old_K.shape[0], device=old_K.device)
            new_K = torch.cat([old_K, cand_K[batch_idx, :, action_idx].unsqueeze(2)], dim=2)
            # new_K: [b, g, s + 1, k]
            
            # old_V: [b, g, s, v]
            # cand_V: [b, g, c, v]
            new_V = torch.cat([old_V, cand_V[batch_idx, :, action_idx].unsqueeze(2)], dim=2)
            # new_V: [b, g, s + 1, v]
            
            new_K_list.append(new_K)
            new_V_list.append(new_V)
        
        batch_idx = torch.arange(cand_room_x.shape[0], device=cand_room_x.device)
        return (
            new_K_list,
            new_V_list,
            cand_room_x[batch_idx, action_idx],
            cand_room_y[batch_idx, action_idx],
            cand_room_placed[batch_idx, action_idx],
        )


    def generate(self, actions: Actions, kv_cache, config: GenerateConfig):
        room_idx = actions.room_idx
        room_x = actions.room_x
        room_y = actions.room_y
        
        n = room_idx.shape[0]  # batch size
        c = room_idx.shape[1]  # number of candidates per batch element
        # e = self.embedding_width
        room_idx = room_idx.to(torch.int64)
        room_x = room_x.to(torch.int64)
        room_y = room_y.to(torch.int64)
        s = kv_cache[0][0].shape[2] if len(kv_cache[0]) > 0 else 0  # current sequence length
        K_list, V_list, old_room_x, old_room_y, old_room_placed = kv_cache
        K_cands = []
        V_cands = []

        with torch.amp.autocast(
            'cuda',
            dtype=torch.bfloat16,
            enabled=room_idx.device.type == 'cuda' and config.state_autocast,
        ):
            X = self.get_embedding(room_idx, room_x, room_y, config)
            # X: [n, c, e]
            # print("generate: X:", X.shape, X)

            for i in range(len(self.attn_layers)):
                h = self.attn_layers[i].num_heads
                g = self.attn_layers[i].num_groups
                k = self.attn_layers[i].key_width
                v = self.attn_layers[i].value_width

                K1 = self.attn_layers[i].key(X)   # [n, c, num_groups * k]
                V1 = self.attn_layers[i].value(X)  # [n, c, num_groups * v]
                K1 = K1.view(n, c, g, k).transpose(1, 2)  # [n, g, c, k]
                V1 = V1.view(n, c, g, v).transpose(1, 2)  # [n, g, c, v]
                K1 = normalize(K1)
                K_cands.append(K1)
                V_cands.append(V1)

                if s > 0:
                    Q = self.attn_layers[i].query(X)
                    Q = Q.view(n, c, h, k).transpose(1, 2)  # [n, h, c, k]
                    Q = normalize(Q)
                    K = K_list[i]  # [n, g, s, k]
                    V = V_list[i]  # [n, g, s, v]
                    A = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
                    # A: [n, h, c, v]
                    A = A.transpose(1, 2).reshape(n, c, h * v)
                    P = self.attn_layers[i].post(A)  # [n, c, e]
                    # print("generate: Q:", Q.shape, Q, "\nK:", K.shape, K, "\nV:", V.shape, V, "\nA:", A.shape, A, "\nP:", P.shape, P)
                    X = X + P
                                                
                X = self.ff_layers[i].forward(X)

            X = normalize(X)
            cand_room_x = old_room_x.unsqueeze(1).expand(-1, c, -1).clone()
            cand_room_y = old_room_y.unsqueeze(1).expand(-1, c, -1).clone()
            cand_room_placed = old_room_placed.unsqueeze(1).expand(-1, c, -1).clone()
            valid_room = room_idx < self.num_rooms
            batch_idx, cand_idx = torch.nonzero(valid_room, as_tuple=True)
            placed_room_idx = room_idx[batch_idx, cand_idx]
            cand_room_x[batch_idx, cand_idx, placed_room_idx] = room_x[batch_idx, cand_idx]
            cand_room_y[batch_idx, cand_idx, placed_room_idx] = room_y[batch_idx, cand_idx]
            cand_room_placed[batch_idx, cand_idx, placed_room_idx] = True
            X = self.get_output(X, cand_room_x, cand_room_y, cand_room_placed)
            X = X.to(torch.float32)

        cache_candidates = (
            K_cands, V_cands, cand_room_x, cand_room_y, cand_room_placed)
        return get_predictions(X, self.output_sizes), cache_candidates


class FrontierStateModel(torch.nn.Module):
    uses_state_features = True

    def __init__(self, num_rooms, output_metadata: OutputMetadata, map_x, map_y, embedding_width, hidden_width, num_layers, frontier_window_size=16, state_features=None, **_):
        super().__init__()
        self.state_features = state_features or {}
        self.num_rooms = num_rooms
        self.map_x = map_x
        self.map_y = map_y
        self.embedding_width = embedding_width
        self.output_sizes = output_metadata.get_output_sizes()
        self.num_connection_outputs = len(output_metadata.connection)
        self.inventory_embedding = torch.nn.Parameter(
            torch.randn([output_metadata.num_room_connection_variants, embedding_width]) / math.sqrt(embedding_width)
        ) if self.state_features.get("inventory", False) else None
        self.orientation_embedding = (
            torch.nn.Embedding(2, embedding_width)
            if self.state_features.get("frontier_orientation", False) else None
        )
        self.kind_embedding = (
            torch.nn.Embedding(256, embedding_width)
            if self.state_features.get("frontier_kind", False) else None
        )
        node_numeric_width = (
            frontier_window_size**2 * self.state_features.get("frontier_occupancy", False)
            + 2 * self.num_connection_outputs * self.state_features.get("frontier_connection_reachability", False)
        )
        self.node_numeric = (
            torch.nn.Linear(node_numeric_width, embedding_width, bias=False)
            if node_numeric_width > 0 else None
        )
        self.frontier_window_area = frontier_window_size**2
        self.register_buffer(
            "frontier_occupancy_bits",
            1 << torch.arange(8, dtype=torch.uint8),
            persistent=False,
        )
        pair_feature_width = (
            embedding_width * self.state_features.get("frontier_neighbor_position", False)
            + 3 * self.state_features.get("frontier_neighbor_flags", False)
        )
        pair_width = 2 * embedding_width + pair_feature_width
        use_neighbors = self.state_features.get("frontier_neighbor", False)
        self.source_message_layers = torch.nn.ModuleList([
            torch.nn.Linear(embedding_width, hidden_width, bias=False)
            for _ in range(num_layers if use_neighbors else 0)
        ])
        self.pair_message_layers = torch.nn.ModuleList([
            torch.nn.Linear(pair_width, hidden_width, bias=False)
            for _ in range(num_layers if use_neighbors else 0)
        ])
        self.message_output_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, embedding_width, bias=False),
            )
            for _ in range(num_layers if use_neighbors else 0)
        ])
        self.update_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(embedding_width * 2, hidden_width, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, embedding_width, bias=False),
            ) for _ in range(num_layers if use_neighbors else 0)
        ])
        global_width = embedding_width * (
            self.state_features.get("inventory", False)
            + 2 * self.state_features.get("frontier_mask", False)
            + (
                self.state_features.get("connection_reachability", False)
                and self.num_connection_outputs > 0
            )
        )
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, embedding_width, bias=False),
        ) if global_width > 0 else None
        self.connection_reachability_embedding = (
            torch.nn.Linear(self.num_connection_outputs, embedding_width, bias=False)
            if self.state_features.get("connection_reachability", False)
            and self.num_connection_outputs > 0 else None
        )
        self.frontier_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
            if self.state_features.get("frontier_position", False) else None
        )
        self.frontier_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
            if self.state_features.get("frontier_position", False) else None
        )
        self.frontier_relative_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
            if self.state_features.get("frontier_neighbor_position", False) else None
        )
        self.frontier_relative_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
            if self.state_features.get("frontier_neighbor_position", False) else None
        )
        self.pos_embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
        self.door_output = FactorizedOutcomeHead(
            output_metadata.door, output_metadata.num_door_variants, embedding_width)
        self.connection_output = FactorizedOutcomeHead(
            output_metadata.connection, output_metadata.num_connection_variants, embedding_width)

    def _position_embedding(self, x, y, embedding_x, embedding_y, offset=0):
        x = x.to(torch.int64) + offset
        y = y.to(torch.int64) + offset
        return embedding_x[x] + embedding_y[y]

    def _pair_features(self, features):
        node = features.frontier
        neighbor = features.frontier_neighbor.clamp_min(0).to(torch.int64)
        def gather_neighbor(values):
            return torch.gather(
                values.unsqueeze(2).expand(-1, -1, neighbor.shape[2]), 1, neighbor
            )
        values = []
        use_position = self.state_features.get("frontier_neighbor_position", False)
        if use_position:
            raw_x = node[:, :, 1].to(torch.int64)
            raw_y = node[:, :, 2].to(torch.int64)
            raw_x0, raw_x1 = raw_x.unsqueeze(2), gather_neighbor(raw_x)
            raw_y0, raw_y1 = raw_y.unsqueeze(2), gather_neighbor(raw_y)
            values.append(self._position_embedding(
                raw_x1 - raw_x0,
                raw_y1 - raw_y0,
                self.frontier_relative_pos_embedding_x,
                self.frontier_relative_pos_embedding_y,
                COORD_OFFSET,
            ))
        if self.state_features.get("frontier_neighbor_flags", False):
            flags = features.frontier_neighbor_pair
            values.append(torch.stack([
                (flags & 1 != 0).to(torch.float32),
                (flags & 2 != 0).to(torch.float32),
                (flags & 4 != 0).to(torch.float32),
            ], dim=-1))
        return torch.cat(values, dim=-1) if values else None

    def forward(self, features: StateFeatures):
        # Shapes below use: b=batch, f=frontiers, k=neighbors, e=embedding width,
        # h=message hidden width.
        # node: [b, f, 5]
        node = features.frontier
        node_mask = node[:, :, 0] != 0
        # numeric: [b, f, numeric_width]
        numeric = []
        if self.state_features.get("frontier_occupancy", False):
            numeric.append(
                features.frontier_occupancy.unsqueeze(-1)
                .bitwise_and(self.frontier_occupancy_bits)
                .ne(0)
                .flatten(-2)[..., :self.frontier_window_area]
                .to(torch.float32)
            )
        if self.state_features.get("frontier_connection_reachability", False):
            flags = features.frontier_connection_reachability
            numeric.append(torch.stack([
                (flags & 1 != 0).to(torch.float32),
                (flags & 2 != 0).to(torch.float32),
            ], dim=-1).flatten(-2))
        # X: [b, f, e]
        X = node.new_zeros([node.shape[0], node.shape[1], self.embedding_width], dtype=torch.float32)
        if self.node_numeric is not None:
            numeric = [value.unsqueeze(-1) if value.ndim == 2 else value for value in numeric]
            X = X + self.node_numeric(torch.cat(numeric, dim=-1))
        if self.frontier_pos_embedding_x is not None:
            X = X + self._position_embedding(
                node[:, :, 1],
                node[:, :, 2],
                self.frontier_pos_embedding_x,
                self.frontier_pos_embedding_y,
            )
        if self.orientation_embedding is not None:
            X = X + self.orientation_embedding(node[:, :, 3].to(torch.int64))
        if self.kind_embedding is not None:
            X = X + self.kind_embedding(node[:, :, 4].to(torch.int64))
        X = X * node_mask.unsqueeze(-1)
        if node.shape[1] == 0:
            mean_pool = max_pool = X.new_zeros([X.shape[0], X.shape[2]])
        else:
            # pair: [b, f, k, pair_feature_width], neighbor: [b, f, k], pair_mask: [b, f, k, 1]
            pair = self._pair_features(features)
            neighbor = features.frontier_neighbor.clamp_min(0).to(torch.int64)
            pair_mask = (features.frontier_neighbor >= 0).unsqueeze(-1)
            for source_layer, pair_layer, output_layer, update_layer in zip(
                self.source_message_layers,
                self.pair_message_layers,
                self.message_output_layers,
                self.update_layers,
            ):
                # Gather each frontier's neighbors: neighbor_state: [b, f, k, e]
                target_state = X.unsqueeze(2).expand(-1, -1, neighbor.shape[2], -1)
                neighbor_state = torch.gather(
                    X,
                    1,
                    neighbor.flatten(1).unsqueeze(-1).expand(-1, -1, X.shape[-1]),
                ).view(*neighbor.shape, X.shape[-1])
                source = source_layer(neighbor_state)
                pair_inputs = [target_state, neighbor_state]
                if pair is not None:
                    pair_inputs.append(pair)
                # messages: [b, f, k, e], then [b, f, e] after neighbor pooling
                messages = source + pair_layer(torch.cat(pair_inputs, dim=-1))
                messages = output_layer(messages) * pair_mask
                messages = messages.sum(2) / pair_mask.sum(2).clamp_min(1)
                X = X + update_layer(torch.cat([X, messages], dim=-1))
                X = X * node_mask.unsqueeze(-1)
            count = node_mask.sum(1, keepdim=True).clamp_min(1)
            mean_pool = X.sum(1) / count
            max_pool = torch.where(node_mask.unsqueeze(-1), X, -torch.inf).max(1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, 0)
        # inventory, mean_pool, max_pool, global_state: [b, e]
        global_inputs = []
        if self.inventory_embedding is not None:
            global_inputs.append(torch.matmul(features.inventory.to(torch.float32), self.inventory_embedding))
        if self.state_features.get("frontier_mask", False):
            global_inputs.extend([mean_pool, max_pool])
        if self.connection_reachability_embedding is not None:
            global_inputs.append(self.connection_reachability_embedding(
                features.connection_reachability.to(torch.float32)
            ))
        global_state = (
            self.global_mlp(torch.cat(global_inputs, dim=-1))
            if self.global_mlp is not None
            else X.new_zeros([X.shape[0], self.embedding_width])
        )
        if self.state_features.get("room_position", False):
            room_x = (features.room_x.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_y = (features.room_y.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_placed = features.room_placed.to(torch.bool).unsqueeze(1)
        else:
            room_x = torch.full([X.shape[0], 1, self.num_rooms], COORD_OFFSET, dtype=torch.int64, device=X.device)
            room_y = room_x
            room_placed = torch.zeros([X.shape[0], 1, self.num_rooms], dtype=torch.bool, device=X.device)
        # X: [b, 1, e]
        X = global_state.unsqueeze(1)
        door = self.door_output(X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y)
        connection = self.connection_output(X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y)
        return get_predictions(torch.cat([door, connection], dim=-1), self.output_sizes)


if __name__ == "__main__":
    rooms = [
        {"map": [[0, 0], [0, 0]], "doors": [[], []], "connections": [[0, 1]]},
        {"map": [[0]], "doors": [[], []], "connections": [[0, 1]]},
        {"map": [[0]], "doors": [], "connections": []},
        {"map": [[0, 0]], "doors": [], "connections": []},
    ]
    state_model = CausalTransformerModel(
        num_rooms=len(rooms),
        output_metadata=OutputMetadata(
            door=[],
            connection=[(0, 0), (1, 1)],
            num_door_variants=0,
            num_connection_variants=2,
            room_connection_variant_idx=[0, 1, 2, 3],
            num_room_connection_variants=4,
        ),
        map_x=8,
        map_y=8,
        embedding_width=3,
        key_width=4,
        value_width=5,
        attn_heads=9,
        head_groups=3,
        hidden_width=7,
        num_layers=2,
    )

    b = 2
    s = 3
    actions = Actions(
        room_idx=torch.randint(0, 4, (b, s)),
        room_x=torch.randint(0, 4, (b, s)),
        room_y=torch.randint(0, 4, (b, s)),
    )
    config = GenerateConfig(
        episode_length=len(rooms),
        max_candidates=4,
        temperature=torch.rand([b]),
    )
    out1 = state_model.forward(actions, config)
    print("forward out:", out1)

    kv_cache = state_model.get_initial_kv_cache(b, "cpu")
    for i in range(s):
        cand = Actions(
            room_idx=actions.room_idx[:, i:i+1],
            room_x=actions.room_x[:, i:i+1],
            room_y=actions.room_y[:, i:i+1],
        )
        out2, kv_cache_cands = state_model.generate(cand, kv_cache, config)
        print(f"generate out {i}:", out2)
        action_idx = torch.zeros(b, dtype=torch.int64)
        kv_cache = state_model.get_updated_kv_cache(kv_cache, kv_cache_cands, action_idx)
