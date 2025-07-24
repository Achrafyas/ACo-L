import copy
import torch
import numpy as np

class Consensus:
    """
    Implements basic consensus operations on model weights.
    """

    def __init__(self):
        pass

    def generate_random(self, n, density=-1):
        """
        Create a random undirected adjacency matrix for n nodes,
        using log(n)/n density by default.
        """
        density = np.log(n) / n if density < 0 else density
        print("network density", density)
        A = np.random.rand(n, n) < density          # random boolean matrix
        A = np.triu(A)                               # keep upper triangle
        A = A + A.T                                  # mirror to lower triangle
        np.fill_diagonal(A, 0)                       # no self-loops
        return A

    def generate_initial_values(self, n, m=1):
        """
        Generate an initial random value vector of shape (n, m).
        """
        return np.random.rand(n, m)

    def laplacian(self, A):
        """
        Compute the graph Laplacian L = D - A,
        where D is the degree matrix.
        """
        D = np.diag(np.sum(A, axis=1))
        return D - A

    def epsilon(self, L):
        """
        Compute a step-size epsilon = 1 / max_degree,
        using the largest diagonal entry of L.
        """
        return 1 / np.max(np.diag(L))

    def perron(self, L):
        """
        Build the Perron (consensus) matrix P = I - epsilon * L.
        """
        n = L.shape[0]
        return np.eye(n) - self.epsilon(L) * L

    def do_consensus(self, A, x0, num_iter):
        """
        Run num_iter iterations of linear consensus:
          x[t] = P x[t-1], starting from x0.
        Returns the final state.
        """
        P = self.perron(self.laplacian(A))
        # allocate array for all iterations
        x = np.zeros(x0.shape + (num_iter,))
        x[..., 0] = x0
        for t in range(1, num_iter):
            x[..., t] = P.dot(x[..., t-1])
        return x[..., -1]

    def apply_consensus(self, own_weights, neighbour_weights, eps):
        """
        Perform one-step pairwise consensus on two sets of PyTorch model weights.
        new = (1 - eps) * own + eps * neighbour, layer by layer.
        """
        # deep copy to avoid mutating input
        updated = copy.deepcopy(own_weights)
        # iterate over each parameter tensor
        for key in own_weights[0].keys():
            # convert to numpy arrays
            own_arr = own_weights[0][key].numpy()
            nbr_arr = neighbour_weights[0][key].numpy()

            # flatten, mix, then reshape
            own_flat = own_arr.flatten()
            nbr_flat = nbr_arr.flatten()
            mix_flat = own_flat + eps * (nbr_flat - own_flat)
            updated[0][key] = torch.from_numpy(mix_flat.reshape(own_arr.shape))

        return updated

    def average_by_consensus(self, w, A, num_iter):
        """
        Compute global average via repeated consensus on each layer across all agents.
        w: list of weight dicts, one per agent.
        A: adjacency matrix.
        num_iter: number of consensus steps.
        Returns a single weight dict for agent 0 after consensus.
        """
        # start from agent 0's weights
        w_avg = copy.deepcopy(w[0])
        # for each parameter tensor
        for key in w_avg.keys():
            # stack each agent's flattened layer into a matrix
            stack = np.stack([agent_w[key].numpy().flatten() for agent_w in w], axis=0)
            # run consensus on this matrix
            result = self.do_consensus(A, stack, num_iter)
            # take the first agent's result and reshape back
            w_avg[key] = torch.from_numpy(result[0].reshape(w_avg[key].shape))
        return w_avg