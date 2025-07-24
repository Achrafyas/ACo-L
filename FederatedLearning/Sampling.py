import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Split MNIST data IID (equal, random) among num_users clients.
    Returns a dict mapping client index to its set of example indices.
    """
    num_items = len(dataset) // num_users  # items per client
    all_idxs = list(range(len(dataset)))
    dict_users = {}
    for i in range(num_users):
        # randomly pick num_items indices without replacement
        chosen = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users[i] = chosen
        # remove chosen indices from pool
        all_idxs = list(set(all_idxs) - chosen)
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Split MNIST data non-IID by sorting labels into shards.
    Each client gets 2 random shards of data (200 shards × 300 images each).
    """
    num_shards, num_imgs = 200, 300
    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort indices by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1].argsort()]
    sorted_idxs = idxs_labels[0]

    # assign 2 shards per client
    for i in range(num_users):
        two_shards = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - two_shards)
        for shard in two_shards:
            start = shard * num_imgs
            end = (shard + 1) * num_imgs
            dict_users[i] = np.concatenate((dict_users[i], sorted_idxs[start:end]))
    return dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Split MNIST data non-IID with unequal shard counts.
    Clients receive a random number of shards (1–30), scaled to total shards.
    """
    num_shards, num_imgs = 1200, 50
    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1].argsort()]
    sorted_idxs = idxs_labels[0]

    # determine random shard sizes per client
    sizes = np.random.randint(1, 31, size=num_users)
    sizes = np.round(sizes / sizes.sum() * num_shards).astype(int)

    # ensure each client gets at least one shard
    for i in range(num_users):
        if not idx_shard:
            break
        take = min(sizes[i], len(idx_shard))
        chosen = set(np.random.choice(idx_shard, take, replace=False))
        idx_shard = list(set(idx_shard) - chosen)
        for shard in chosen:
            start = shard * num_imgs
            end = (shard + 1) * num_imgs
            dict_users[i] = np.concatenate((dict_users[i], sorted_idxs[start:end]))

    # if leftovers remain, assign to client with fewest samples
    if idx_shard:
        remaining = idx_shard
        k = min(dict_users, key=lambda x: len(dict_users[x]))
        for shard in remaining:
            start = shard * num_imgs
            end = (shard + 1) * num_imgs
            dict_users[k] = np.concatenate((dict_users[k], sorted_idxs[start:end]))
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Split CIFAR data IID among clients.
    Similar to mnist_iid.
    """
    num_items = len(dataset) // num_users
    all_idxs = list(range(len(dataset)))
    dict_users = {}
    for i in range(num_users):
        chosen = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users[i] = chosen
        all_idxs = list(set(all_idxs) - chosen)
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Split CIFAR data non-IID by label-based shards (200 shards × 250 images).
    Each client gets 2 random shards.
    """
    num_shards, num_imgs = 200, 250
    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.train_labels)

    # sort by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1].argsort()]
    sorted_idxs = idxs_labels[0]

    # assign 2 shards per client
    for i in range(num_users):
        two_shards = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - two_shards)
        for shard in two_shards:
            start = shard * num_imgs
            end = (shard + 1) * num_imgs
            dict_users[i] = np.concatenate((dict_users[i], sorted_idxs[start:end]))
    return dict_users

if __name__ == '__main__':
    # Example usage: test non-IID split
    from torchvision import datasets, transforms
    ds = datasets.MNIST(
        './data/mnist/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    # Generate non-IID split for 100 clients
    user_data = mnist_noniid(ds, 100)
    print("Client 0 has", len(user_data[0]), "samples")
