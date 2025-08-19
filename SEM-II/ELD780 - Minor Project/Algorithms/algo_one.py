import numpy as np

class Client:
    def __init__(self, data_x, data_y, learning_rate):
        self.data_x = data_x
        self.data_y = data_y
        self.learning_rate = learning_rate

    def train(self, global_model, steps):
        local_model = np.copy(global_model)
        for _ in range(steps):
            grad = self.compute_gradient(local_model)
            local_model -= self.learning_rate * grad
        return local_model

    def compute_gradient(self, model):
        predictions = self.data_x.dot(model)
        error = predictions - self.data_y
        return self.data_x.T.dot(error) / len(self.data_y)

class Server:
    def __init__(self, clients, initial_model, nu=1e-6, aggregation_iters=3):
        self.clients = clients
        self.global_model = initial_model
        self.nu = nu
        self.aggregation_iters = aggregation_iters

    def select_clients(self, num_clients):
        return np.random.choice(self.clients, num_clients, replace=False)

    def aggregate(self, updates, alphas):
        updates = np.array(updates)
        alphas = np.array(alphas)
        v = np.average(updates, axis=0, weights=alphas)
        for _ in range(self.aggregation_iters):
            distances = np.linalg.norm(updates - v, axis=1)
            weights = alphas / np.maximum(self.nu, distances)
            weights /= weights.sum()
            v = np.sum(weights[:, np.newaxis] * updates, axis=0)
        return v

    def train_round(self, clients_per_round, local_steps):
        """Complete one round of federated training"""
        selected_clients = self.select_clients(clients_per_round)
        updates = []
        alphas = []
        for client in selected_clients:
            local_model = client.train(self.global_model, local_steps)
            updates.append(local_model)
            alphas.append(len(client.data_y))
        alphas = np.array(alphas) / sum(alphas)
        self.global_model = self.aggregate(updates, alphas)

def generate_synthetic_data(num_clients=10, model_dim=5, data_std=0.1):
    true_model = np.random.randn(model_dim)
    clients = []
    for _ in range(num_clients):
        n_samples = np.random.randint(50, 200)
        X = np.random.randn(n_samples, model_dim)
        y = X.dot(true_model) + data_std * np.random.randn(n_samples)
        clients.append(Client(X, y, learning_rate=0.01))
    return clients, true_model

def main():
    model_dim = 5
    num_clients = 20
    rounds = 100
    clients_per_round = 5
    local_steps = 10
    
    clients, true_model = generate_synthetic_data(num_clients, model_dim)
    server = Server(clients, initial_model=np.zeros(model_dim))

    for r in range(rounds):
        server.train_round(clients_per_round, local_steps)
        error = np.linalg.norm(server.global_model - true_model)
        if r % 10 == 0:
            print(f"Round {r:3d} | Parameter error: {error:.4f}")

if __name__ == "__main__":
    main()