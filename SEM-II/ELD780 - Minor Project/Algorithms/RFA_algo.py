import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# --------------------------
# Core Algorithm Implementations
# --------------------------

class RFAAggregator:
    """Implements all aggregation strategies from the paper"""

    @staticmethod
    def fedavg(updates, weights):
        return np.average(updates, axis=0, weights=weights)

    @staticmethod
    def smoothed_weiszfeld(updates, weights, nu=1e-6, max_iter=100, tol=1e-6):
        updates = np.array(updates)
        weights = np.array(weights) / sum(weights)
        v = np.average(updates, axis=0, weights=weights)

        for _ in range(max_iter):
            residuals = updates - v
            distances = np.linalg.norm(residuals, axis=1)
            weights_iter = weights / np.maximum(nu, distances)
            weights_iter /= weights_iter.sum()
            new_v = np.sum(weights_iter[:, None] * updates, axis=0)
            if np.linalg.norm(new_v - v) < tol:
                break
            v = new_v

        return v

    @staticmethod
    def one_step_rfa(updates, weights, nu=1e-6):
        norms = np.linalg.norm(updates, axis=1)
        weights = weights / np.maximum(nu, norms)
        weights /= weights.sum()
        return np.average(updates, axis=0, weights=weights)

    @staticmethod
    def personalized_aggregate(global_updates, weights, nu=1e-6):
        return RFAAggregator.smoothed_weiszfeld(global_updates, weights, nu)

# --------------------------
# Client Implementations
# --------------------------

class BaseClient(ABC):
    def __init__(self, client_id, data, model_dim, learning_rate):
        self.id = client_id
        self.X, self.y = data
        self.model_dim = model_dim
        self.lr = learning_rate

    @abstractmethod
    def compute_update(self, global_model):
        pass

class StandardClient(BaseClient):
    def compute_update(self, global_model):
        w = global_model.copy()
        self.local_steps = 5
        for _ in range(self.local_steps):
            grad = self._compute_gradient(w)
            w -= self.lr * grad
        return w

    def _compute_gradient(self, model):
        return self.X.T @ (self.X @ model - self.y) / len(self.y)

class PersonalizedClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_params = np.random.randn(self.model_dim) * 0.01

    def compute_update(self, global_model):
        self.personal_steps = 5
        self.global_steps = 5

        combined_model = global_model + self.personal_params
        for _ in range(self.personal_steps):
            grad = self._compute_gradient(combined_model)
            self.personal_params -= self.lr * grad

        global_grads = []
        current_global = global_model.copy()
        for _ in range(self.global_steps):
            combined_model = current_global + self.personal_params
            grad = self._compute_gradient(combined_model)
            current_global -= self.lr * grad
            global_grads.append(grad)

        return np.mean(global_grads, axis=0)

    def _compute_gradient(self, model):
        return self.X.T @ (self.X @ model - self.y) / len(self.y)

class CorruptedClient(StandardClient):
    def __init__(self, corruption_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption_type = corruption_type

    def compute_update(self, global_model):
        clean_update = super().compute_update(global_model)
        if self.corruption_type == "random":
            return np.random.randn(*clean_update.shape)
        elif self.corruption_type == "inverted":
            return -clean_update
        elif self.corruption_type == "gaussian":
            return clean_update + np.random.randn(*clean_update.shape) * np.std(clean_update)
        return clean_update

# --------------------------
# Server Implementation
# --------------------------

class RFAServer:
    def __init__(self, model_dim, aggregator='rfa', nu=1e-6):
        self.global_model = np.zeros(model_dim)
        self.aggregator = aggregator
        self.nu = nu
        self.history = []

    def aggregate_updates(self, updates, weights):
        if self.aggregator == 'fedavg':
            return RFAAggregator.fedavg(updates, weights)
        elif self.aggregator == 'rfa':
            return RFAAggregator.smoothed_weiszfeld(updates, weights, self.nu)
        elif self.aggregator == 'one-step':
            return RFAAggregator.one_step_rfa(updates, weights, self.nu)
        elif self.aggregator == 'personalized':
            return RFAAggregator.personalized_aggregate(updates, weights, self.nu)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def train_round(self, clients, corruption_level=0.0):
        updates = []
        weights = []
        corrupted_indices = []

        num_corrupted = int(len(clients) * corruption_level)
        corrupt_indices = np.random.choice(len(clients), num_corrupted, replace=False)

        for i, client in enumerate(clients):
            if i in corrupt_indices and isinstance(client, CorruptedClient):
                update = client.compute_update(self.global_model)
                corrupted_indices.append(i)
            else:
                update = client.compute_update(self.global_model)

            updates.append(update)
            weights.append(len(client.y))

        weights = np.array(weights) / sum(weights)
        self.global_model = self.aggregate_updates(updates, weights)

        self.history.append({
            'updates': updates,
            'weights': weights,
            'corrupted': corrupted_indices
        })

# --------------------------
# Synthetic Data Generation
# --------------------------

def generate_synthetic_data(num_clients, model_dim, corruption=0.0, personalization=0.0):
    base_model = np.random.randn(model_dim)
    clients = []

    for i in range(num_clients):
        n_samples = np.random.randint(50, 200)
        X = np.random.randn(n_samples, model_dim)

        if personalization > 0 and np.random.rand() < 0.5:
            true_model = base_model + np.random.randn(model_dim) * personalization
        else:
            true_model = base_model

        y = X @ true_model + np.random.normal(0, 0.1, n_samples)

        if np.random.rand() < corruption:
            clients.append(CorruptedClient(
                corruption_type=np.random.choice(["random", "inverted", "gaussian"]),
                client_id=i,
                data=(X, y),
                model_dim=model_dim,
                learning_rate=0.01
            ))
        else:
            clients.append(StandardClient(
                client_id=i,
                data=(X, y),
                model_dim=model_dim,
                learning_rate=0.01
            ))

    return clients, base_model

# --------------------------
# Experiment Framework
# --------------------------

def run_experiment(config):
    clients, true_model = generate_synthetic_data(
        config['num_clients'],
        config['model_dim'],
        corruption=config['corruption_level'],
        personalization=config['personalization_strength']
    )

    servers = {
        alg: RFAServer(config['model_dim'], aggregator=alg)
        for alg in ['fedavg', 'rfa', 'one-step', 'personalized']
    }

    results = {alg: [] for alg in servers}

    for round in range(config['num_rounds']):
        for alg, server in servers.items():
            server.train_round(clients, config['corruption_level'])
            error = np.linalg.norm(server.global_model - true_model)
            results[alg].append(error)

        if round % 10 == 0:
            errors_str = {alg: f"{results[alg][-1]:.4f}" for alg in servers}
            print(f"Round {round:3d} | Errors: {errors_str}")

    plt.figure(figsize=(10, 6))
    for alg, errors in results.items():
        plt.plot(errors, label=alg, linewidth=2)

    plt.title("RFA Algorithm Comparison (Paper Reproduction)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Model Error (L2 Distance from True Model)")
    plt.grid(True)
    plt.legend()
    plt.show()

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    experiment_config = {
        'num_clients': 100,
        'model_dim': 20,
        'num_rounds': 100,
        'corruption_level': 0.3,
        'personalization_strength': 0.2
    }
    run_experiment(experiment_config)