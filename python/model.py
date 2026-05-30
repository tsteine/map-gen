import torch
import math
from dataclasses import dataclass

from env import Actions, GenerateConfig

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


def get_outcome_metadata(rooms):
    direction_order = {"left": 0, "right": 1, "up": 2, "down": 3}
    geometry_by_key = {}
    geometry_idx_by_room = []
    geometry_doors = []
    geometry_connections = []

    for room in rooms:
        doors = sorted(
            (
                door["direction"],
                door["x"],
                door["y"],
                door.get("kind", 0),
            )
            for door_group in room.get("doors", [])
            for door in door_group
        )
        geometry_key = (
            tuple(tuple(row) for row in room["map"]),
            tuple(sorted(doors, key=lambda door: (direction_order[door[0]], *door[1:]))),
        )
        geometry_idx = geometry_by_key.get(geometry_key)
        if geometry_idx is None:
            geometry_idx = len(geometry_by_key)
            geometry_by_key[geometry_key] = geometry_idx
            geometry_doors.append(geometry_key[1])
            geometry_connections.append(set())
        geometry_idx_by_room.append(geometry_idx)
        geometry_connections[geometry_idx].update(tuple(connection) for connection in room.get("connections", []))

    geometry_connections = [sorted(connections) for connections in geometry_connections]
    door_identity = [
        {door: identity_idx for identity_idx, door in enumerate(doors)}
        for doors in geometry_doors
    ]
    connection_identity = [
        {connection: identity_idx for identity_idx, connection in enumerate(connections)}
        for connections in geometry_connections
    ]

    door_outputs = []
    for direction in direction_order:
        for room_idx, room in enumerate(rooms):
            geometry_idx = geometry_idx_by_room[room_idx]
            for door_group in room.get("doors", []):
                for door in door_group:
                    if door["direction"] == direction:
                        door_key = (direction, door["x"], door["y"], door.get("kind", 0))
                        door_outputs.append((room_idx, geometry_idx, door_identity[geometry_idx][door_key]))

    connection_outputs = []
    for room_idx, room in enumerate(rooms):
        geometry_idx = geometry_idx_by_room[room_idx]
        for connection in room.get("connections", []):
            connection_outputs.append(
                (room_idx, geometry_idx, connection_identity[geometry_idx][tuple(connection)])
            )

    return (
        len(geometry_by_key),
        door_outputs,
        connection_outputs,
        max((len(doors) for doors in geometry_doors), default=0),
        max((len(connections) for connections in geometry_connections), default=0),
    )


class FactorizedOutcomeHead(torch.nn.Module):
    def __init__(self, output_metadata, num_geometries, num_identities, embedding_width):
        super().__init__()
        self.embedding_width = embedding_width
        self.num_outputs = len(output_metadata)
        metadata = torch.tensor(output_metadata, dtype=torch.int64).reshape(self.num_outputs, 3)
        self.register_buffer("room_idx", metadata[:, 0])
        self.register_buffer("geometry_idx", metadata[:, 1])
        self.register_buffer("identity_idx", metadata[:, 2])
        self.geometry_embedding = torch.nn.Parameter(
            torch.randn([num_geometries, embedding_width]) / math.sqrt(embedding_width))
        self.identity_embedding = torch.nn.Parameter(
            torch.randn([num_identities, embedding_width]) / math.sqrt(embedding_width))
        self.state = torch.nn.Linear(embedding_width, embedding_width, bias=False)

    def forward(self, X, room_x, room_y, room_placed, pos_embedding_x, pos_embedding_y):
        if self.num_outputs == 0:
            return X.new_empty([X.shape[0], X.shape[1], 0])
        state = self.state(X)
        base_query = self.geometry_embedding[self.geometry_idx] + self.identity_embedding[self.identity_idx]
        base_logits = torch.matmul(state, base_query.transpose(0, 1))
        x_logits = torch.matmul(state, pos_embedding_x.transpose(0, 1))
        y_logits = torch.matmul(state, pos_embedding_y.transpose(0, 1))
        room_logits = torch.gather(x_logits, -1, room_x) + torch.gather(y_logits, -1, room_y)
        room_logits = torch.where(room_placed, room_logits, 0.0)
        position_logits = room_logits[..., self.room_idx]
        return (base_logits + position_logits) / math.sqrt(self.embedding_width)


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
    def __init__(self, rooms, map_x, map_y, output_sizes, embedding_width, key_width, value_width, attn_heads, head_groups, hidden_width, num_layers):
        super().__init__()
        self.num_rooms = len(rooms)
        self.map_x = map_x
        self.map_y = map_y
        self.num_tokens = self.num_rooms + 1
        self.output_sizes = output_sizes
        self.num_outputs = sum(output_sizes)
        self.num_layers = num_layers
        self.embedding_width = embedding_width
        self.global_lin = torch.nn.Linear(1, embedding_width)
        self.pos_embedding_x = torch.nn.Parameter(torch.randn([self.map_x, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_y = torch.nn.Parameter(torch.randn([self.map_y, embedding_width]) / math.sqrt(embedding_width))
        self.room_embedding = torch.nn.Parameter(
            torch.randn([self.num_rooms + 1, embedding_width]) / math.sqrt(embedding_width))
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

        num_geometries, door_outputs, connection_outputs, num_door_identities, num_connection_identities = (
            get_outcome_metadata(rooms)
        )
        assert output_sizes == (len(door_outputs), len(connection_outputs))
        self.door_output = FactorizedOutcomeHead(
            door_outputs, num_geometries, num_door_identities, embedding_width)
        self.connection_output = FactorizedOutcomeHead(
            connection_outputs, num_geometries, num_connection_identities, embedding_width)


    def get_embedding(self, room_idx, room_x, room_y, config: GenerateConfig):
        # global_data = torch.cat([torch.log(config.temperature.view(-1, 1))], dim=1)

        # global_emb = self.global_lin(global_data).unsqueeze(1)
        # TODO: try rotary positional embeddings
        position_emb_x = self.pos_embedding_x[room_x]
        position_emb_y = self.pos_embedding_y[room_y]
        room_emb = self.room_embedding[room_idx]
        
        # X = global_emb + position_emb_x + position_emb_y + room_emb
        X = position_emb_x + position_emb_y + room_emb
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
            X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y)
        connection = self.connection_output(
            X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y)
        return torch.cat([door, connection], dim=-1)


    def forward(self, actions: Actions, config: GenerateConfig):
        room_idx = actions.room_idx.to(torch.int64)
        room_x = actions.room_x.to(torch.int64)
        room_y = actions.room_y.to(torch.int64)

        with torch.amp.autocast('cuda', enabled=room_idx.device.type == 'cuda'):
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

        with torch.amp.autocast('cuda', enabled=room_idx.device.type == 'cuda'):
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


if __name__ == "__main__":
    rooms = [
        {"map": [[0, 0], [0, 0]], "doors": [[], []], "connections": [[0, 1]]},
        {"map": [[0]], "doors": [[], []], "connections": [[0, 1]]},
        {"map": [[0]], "doors": [], "connections": []},
        {"map": [[0, 0]], "doors": [], "connections": []},
    ]
    state_model = CausalTransformerModel(
        rooms=rooms,
        map_x=8,
        map_y=8,
        output_sizes=(0, 2),
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
