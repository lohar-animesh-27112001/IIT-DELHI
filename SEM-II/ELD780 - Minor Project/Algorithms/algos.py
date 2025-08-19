import numpy as np
from abc import ABC, abstractmethod

class BaseClient(ABC):
    def __init__(self, data_x, data_y, model_dim, learning_rate):
        self.data_x = data_x
        self.data_y = data_y
        self.model_dim = model_dim
        self.learning_rate = learning_rate
        
    @abstractmethod
    def train(self, global_model, steps):
        pass

    def compute_gradient(self, model):
        """Compute MSE gradient for linear regression"""
        predictions = self.data_x.dot(model)
        error = predictions - self.data_y
        return self.data_x.T.dot(error) / len(self.data_y)

class StandardClient(BaseClient):
    def train(self, global_model, steps):
        """Vanilla federated learning client"""
        model = np.copy(global_model)
        for _ in range(steps):
            model -= self.learning_rate * self.compute_gradient(model)
        return model

class PersonalizedClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_params = np.random.randn(self.model_dim) * 0.1
        
    def train(self, global_model, steps):
        """Personalized federated learning client"""
        # Train personal parameters
        combined_model = global_model + self.personal_params
        for _ in range(steps):
            grad = self.compute_gradient(combined_model)
            self.personal_params -= self.learning_rate * grad
        
        # Train global parameters
        global_grads = []
        current_global = global_model.copy()
        for _ in range(steps):
            combined_model = current_global + self.personal_params
            grad = self.compute_gradient(combined_model)
            current_global -= self.learning_rate * grad
            global_grads.append(grad)
            
        return np.mean(global_grads, axis=0)

class FederatedServer:
    AGGREGATORS = {
        'fedavg': lambda s, u, a: np.average(u, axis=0, weights=a),
        'rfa': lambda s, u, a: s.geometric_median(u, a),
        'one-step': lambda s, u, a: s.one_step(u, a),
        'personalized': lambda s, u, a: s.personalized_aggregate(u, a)
    }
    
    def __init__(self, clients, model_dim, aggregator='fedavg', nu=1e-6, max_iter=3):
        self.clients = clients
        self.global_model = np.zeros(model_dim)
        self.aggregator = aggregator
        self.nu = nu
        self.max_iter = max_iter
        
    def select_clients(self, num_clients):
        return np.random.choice(self.clients, num_clients, replace=False)
    
    def geometric_median(self, updates, alphas):
        """Algorithm 1 & 2: Smoothed Weiszfeld"""
        updates = np.array(updates)
        alphas = np.array(alphas) / sum(alphas)
        estimate = np.average(updates, axis=0, weights=alphas)
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(updates - estimate, axis=1)
            weights = alphas / np.maximum(self.nu, distances)
            estimate = np.sum(updates * weights[:, None], axis=0) / weights.sum()
            
        return estimate
    
    def one_step(self, updates, alphas):
        """Algorithm 3: One-Step RFA"""
        norms = np.linalg.norm(updates, axis=1)
        weights = alphas / np.maximum(self.nu, norms)
        return np.average(updates, axis=0, weights=weights)
    
    def personalized_aggregate(self, updates, alphas):
        """Algorithm 4: Personalized RFA"""
        # For personalized, updates contain both global and personal components
        return self.geometric_median(updates, alphas)
    
    def train_round(self, clients_per_round, local_steps):
        selected = self.select_clients(clients_per_round)
        updates = []
        alphas = []
        
        for client in selected:
            if isinstance(client, PersonalizedClient):
                update = client.train(self.global_model, local_steps)
            else:
                update = client.train(self.global_model, local_steps)
                
            updates.append(update)
            alphas.append(len(client.data_x))
        
        agg_fn = self.AGGREGATORS[self.aggregator]
        self.global_model = agg_fn(self, updates, alphas)

def generate_data(num_clients=10, model_dim=5, personal_std=0.3, data_std=0.1):
    """Generate synthetic dataset with optional personalization"""
    base_model = np.random.randn(model_dim)
    clients = []
    
    for _ in range(num_clients):
        n_samples = np.random.randint(50, 200)
        X = np.random.randn(n_samples, model_dim)
        
        if personal_std > 0:
            personal_model = np.random.randn(model_dim) * personal_std
            y = X.dot(base_model + personal_model) + data_std * np.random.randn(n_samples)
            clients.append(PersonalizedClient(X, y, model_dim, 0.01))
        else:
            y = X.dot(base_model) + data_std * np.random.randn(n_samples)
            clients.append(StandardClient(X, y, model_dim, 0.01))
    
    return clients, base_model

def main():
    # Configuration
    config = {
        'model_dim': 5,
        'num_clients': 20,
        'rounds': 100,
        'clients_per_round': 5,
        'local_steps': 10,
        'aggregator': 'rfa',  # Choose from: fedavg, rfa, one-step, personalized
        'personal_std': 0.3
    }
    
    # Initialize
    clients, true_model = generate_data(
        config['num_clients'], 
        config['model_dim'],
        config['personal_std']
    )
    
    server = FederatedServer(
        clients,
        config['model_dim'],
        aggregator=config['aggregator']
    )
    
    # Training loop
    errors = []
    for r in range(config['rounds']):
        server.train_round(config['clients_per_round'], config['local_steps'])
        error = np.linalg.norm(server.global_model - true_model)
        errors.append(error)
        
        if r % 10 == 0:
            print(f"Round {r:3d} | Error: {error:.4f}")
    
    return errors

def compare_algorithms():
    """Compare all algorithms performance"""
    config = {
        'model_dim': 5,
        'num_clients': 20,
        'rounds': 50,
        'clients_per_round': 5,
        'local_steps': 10,
        'personal_std': 0.5
    }
    
    results = {}
    for alg in ['fedavg', 'rfa', 'one-step', 'personalized']:
        print(f"\nRunning {alg}...")
        clients, true_model = generate_data(
            config['num_clients'], 
            config['model_dim'],
            config['personal_std']
        )
        server = FederatedServer(clients, config['model_dim'], aggregator=alg)
        
        errors = []
        for r in range(config['rounds']):
            server.train_round(config['clients_per_round'], config['local_steps'])
            errors.append(np.linalg.norm(server.global_model - true_model))
        
        results[alg] = errors
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for alg, errors in results.items():
        plt.plot(errors, label=alg)
    plt.xlabel('Rounds')
    plt.ylabel('Model Error')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run single algorithm
    # main()
    
    # Compare all algorithms
    compare_algorithms()