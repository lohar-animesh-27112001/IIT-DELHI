import numpy as np

class PersonalizedClient:
    def __init__(self, data_x, data_y, model_dim, learning_rate):
        self.data_x = data_x
        self.data_y = data_y
        self.learning_rate = learning_rate
        self.personal_params = np.random.randn(model_dim) * 0.1
        self.global_model = np.zeros(model_dim)

    def train_personal(self, steps):
        for _ in range(steps):
            combined_model = self.global_model + self.personal_params
            grad = self.compute_gradient(combined_model)
            self.personal_params -= self.learning_rate * grad

    def train_global(self, steps):
        global_grads = []
        current_global = self.global_model.copy()
        for _ in range(steps):
            combined_model = current_global + self.personal_params
            grad = self.compute_gradient(combined_model)
            current_global -= self.learning_rate * grad
            global_grads.append(grad)
        return np.mean(global_grads, axis=0)

    def compute_gradient(self, model):
        predictions = self.data_x.dot(model)
        error = predictions - self.data_y
        return self.data_x.T.dot(error) / len(self.data_y)

class PersonalizedRFAServer:
    def __init__(self, clients, model_dim, nu=1e-6, aggregation_iters=3):
        self.clients = clients
        self.global_model = np.zeros(model_dim)
        self.nu = nu
        self.aggregation_iters = aggregation_iters

    def select_clients(self, num_clients):
        return np.random.choice(self.clients, num_clients, replace=False)

    def robust_aggregate(self, updates, alphas):
        updates = np.array(updates)
        alphas = np.array(alphas, dtype=np.float64)
        alphas /= alphas.sum()
        estimate = np.average(updates, axis=0, weights=alphas)
        for _ in range(self.aggregation_iters):
            residuals = updates - estimate
            distances = np.linalg.norm(residuals, axis=1)
            weights = alphas / np.maximum(self.nu, distances)
            weights /= weights.sum()
            estimate = np.sum(weights[:, None] * updates, axis=0)

        return estimate

    def train_round(self, clients_per_round, personal_steps, global_steps):
        selected_clients = self.select_clients(clients_per_round)
        updates = []
        alphas = []
        for client in selected_clients:
            client.global_model = self.global_model.copy()
            client.train_personal(personal_steps)
        for client in selected_clients:
            global_update = client.train_global(global_steps)
            updates.append(global_update)
            alphas.append(len(client.data_y))
        self.global_model += self.robust_aggregate(updates, alphas)

def generate_synthetic_data(num_clients=10, model_dim=5, data_std=0.1, personal_std=0.3):
    base_model = np.random.randn(model_dim)
    clients = []
    
    for _ in range(num_clients):
        personal_model = np.random.randn(model_dim) * personal_std
        n_samples = np.random.randint(50, 200)
        X = np.random.randn(n_samples, model_dim)
        y = X.dot(base_model + personal_model) + data_std * np.random.randn(n_samples)
        clients.append(PersonalizedClient(X, y, model_dim, learning_rate=0.01))
    return clients, base_model

def main():
    model_dim = 5
    num_clients = 20
    rounds = 100
    clients_per_round = 5
    personal_steps = 5
    global_steps = 10
    clients, true_global = generate_synthetic_data(num_clients, model_dim)
    server = PersonalizedRFAServer(clients, model_dim)
    for r in range(rounds):
        server.train_round(clients_per_round, personal_steps, global_steps)
        global_error = np.linalg.norm(server.global_model - true_global)
        personal_errors = [np.linalg.norm(c.personal_params) for c in clients]
        if r % 10 == 0:
            print(f"Round {r:3d} | Global error: {global_error:.4f} | "
                  f"Avg personal norm: {np.mean(personal_errors):.4f}")

if __name__ == "__main__":
    main()