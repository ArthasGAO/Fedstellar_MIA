from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset, ConcatDataset, Dataset, DataLoader


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx


class FedstellarDataset(Dataset, ABC):
    """
    Abstract class for a partitioned dataset.

    Classes inheriting from this class need to implement specific methods
    for loading and partitioning the dataset.
    """

    def __init__(
            self,
            num_classes=10,
            sub_id=0,
            number_sub=1,
            batch_size=32,
            num_workers=4,
            val_percent=0.1,
            iid=True,
            partition="dirichlet",
            seed=42,
            config=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.iid = iid
        self.partition = partition
        self.seed = seed
        self.config = config

        self.train_set = None
        self.train_indices_map = None
        self.test_set = None
        self.test_indices_map = None

        # MIA setting
        self.indexing_map = None  # this is used to decompose the MIA result from micro view for the in eval group.

        self.shadow_train = None
        self.shadow_test = None
        self.in_eval = None
        self.out_eval = None

        self.initialize_dataset()

    @abstractmethod
    def initialize_dataset(self):
        """
        Initialize the dataset. This should load or create the dataset.
        """
        pass

    @abstractmethod
    def generate_non_iid_map(self, dataset, partition="dirichlet"):
        """
        Create a non-iid map of the dataset.
        """
        pass

    @abstractmethod
    def generate_iid_map(self, dataset):
        """
        Create an iid map of the dataset.
        """
        pass

    def plot_data_distribution(self, dataset, partitions_map):
        """
        Plot the data distribution of the dataset.

        Plot the data distribution of the dataset according to the partitions map provided.

        Args:
            dataset: The dataset to plot (torch.utils.data.Dataset).
            partitions_map: The map of the dataset partitions.
        """
        # Plot the data distribution of the dataset, one graph per partition
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")

        for i in range(self.number_sub):
            indices = partitions_map[i]
            class_counts = [0] * self.num_classes
            for idx in indices:
                _, label = dataset[idx]
                class_counts[label] += 1
            print(f"Participant {i + 1} class distribution: {class_counts}")
            plt.figure()
            plt.bar(range(self.num_classes), class_counts)
            plt.xlabel("Class")
            plt.ylabel("Number of samples")
            plt.xticks(range(self.num_classes))
            plt.title(
                f"Partition {i + 1} class distribution {'(IID)' if self.iid else '(Non-IID)'}{' - ' + self.partition if not self.iid else ''}")
            plt.tight_layout()
            path_to_save = f"{self.config.participant['tracking_args']['log_dir']}/{self.config.participant['scenario_args']['name']}/participant_{i + 1}_class_distribution_{'iid' if self.iid else 'non_iid'}{'_' + self.partition if not self.iid else ''}.png"
            plt.savefig(
                path_to_save, dpi=300, bbox_inches="tight"
            )
            plt.close()

    def dirichlet_partition(self, dataset, alpha=0.5):
        """
        Partition the dataset into multiple subsets using a Dirichlet distribution.

        This function divides a dataset into a specified number of subsets (federated clients),
        where each subset has a different class distribution. The class distribution in each
        subset is determined by a Dirichlet distribution, making the partition suitable for 
        simulating non-IID (non-Independently and Identically Distributed) data scenarios in 
        federated learning.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have 
                                                'data' and 'targets' attributes.
            alpha (float): The concentration parameter of the Dirichlet distribution. A lower 
                        alpha value leads to more imbalanced partitions.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to number_sub-1) 
                and values are lists of indices corresponding to the samples in each subset.

        The function ensures that each class is represented in each subset but with varying 
        proportions. The partitioning process involves iterating over each class, shuffling 
        the indices of that class, and then splitting them according to the Dirichlet 
        distribution. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = dirichlet_partition(my_dataset, alpha=0.5)
            # This creates federated data subsets with varying class distributions based on
            # a Dirichlet distribution with alpha = 0.5.
        """
        np.random.seed(self.seed)
        X_train, y_train = dataset.data, np.array(dataset.targets)
        min_size = 0
        K = self.num_classes
        N = y_train.shape[0]
        n_nets = self.number_sub
        node_size = int(self.config.participant["mia_args"]["data_size"])
        net_dataidx_map = {}

        if node_size:
            restricted_size = n_nets * node_size
            idxs = np.random.permutation(N)[:restricted_size]
            out_idxs = np.random.permutation(N)[restricted_size:]
            X_train, y_train = X_train[idxs], y_train[idxs]
            N = restricted_size

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # partitioned_datasets = []
        for i in range(self.number_sub):
            #    subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            #    partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                _, label = dataset[idx]
                class_counts[label] += 1
            # print(f"Partition {i+1} class distribution: {class_counts}")

        total_size = sum(len(indices) for indices in net_dataidx_map.values())
        if total_size != N:
            raise ValueError(f"Total size of partitioned data {total_size} does not match restricted size {N}.")

        if self.config.participant["mia_args"]["attack_type"] != "No Attack":
            self.initialize_eval_dataset(idxs, out_idxs)
            if self.config.participant["mia_args"]["attack_type"] == "Shadow Model Based MIA" \
                    or self.config.participant["mia_args"]["metric_detail"] in {"Prediction Class Confidence",
                                                                                "Prediction Class Entropy",
                                                                                "Prediction Modified Entropy"}:
                self.initialize_shadow_dataset(out_idxs, node_size * n_nets,
                                               self.config.participant["mia_args"]["shadow_model_number"])

        self.indexing_map = net_dataidx_map

        return net_dataidx_map

    def homo_partition(self, dataset):
        """
        Homogeneously partition the dataset into multiple subsets.

        This function divides a dataset into a specified number of subsets, where each subset
        is intended to have a roughly equal number of samples. This method aims to ensure a
        homogeneous distribution of data across all subsets. It's particularly useful in 
        scenarios where a uniform distribution of data is desired among all federated learning 
        clients.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have 
                                                'data' and 'targets' attributes.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to number_sub-1) 
                and values are lists of indices corresponding to the samples in each subset.

        The function randomly shuffles the entire dataset and then splits it into the number 
        of subsets specified by `number_sub`. It ensures that each subset has a similar number
        of samples. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = homo_partition(my_dataset)
            # This creates federated data subsets with homogeneous distribution.
        """
        np.random.seed(self.seed)
        n_nets = self.number_sub
        node_size = int(self.config.participant["mia_args"]["data_size"])

        '''print(type(node_size))
        print(type(n_nets))
        print(node_size)
        print(n_nets)'''

        n_train = dataset.data.shape[0]
        idxs = np.random.permutation(n_train)
        in_idxs, out_idxs = None, None

        if node_size:
            restricted_size = n_nets * node_size
            in_dxs = idxs[:restricted_size]
            if n_train >= 2 * restricted_size:
                out_idxs = idxs[restricted_size:2 * restricted_size]
            else:
                print("""
                Warning: The out evaluation dataset is not enough to match the in evaluation dataset size.
                You may want to reconsider the evaluation of the precision of MIA here.
                """)
                out_idxs = idxs[restricted_size:]

        '''print(len(in_dxs))
        print(len(out_idxs))
        print(len(idxs))'''

        batch_idxs = np.array_split(in_dxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        # partitioned_datasets = []
        for i in range(self.number_sub):
            # subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            # partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                _, label = dataset[idx]
                class_counts[label] += 1
            print(f"Partition {i + 1} class distribution: {class_counts}")

        if self.config.participant["mia_args"]["attack_type"] != "No Attack":
            self.initialize_eval_dataset(in_dxs, out_idxs)
            if self.config.participant["mia_args"]["attack_type"] == "Shadow Model Based MIA" \
                    or self.config.participant["mia_args"]["metric_detail"] in {"Prediction Class Confidence",
                                                                                "Prediction Class Entropy",
                                                                                "Prediction Modified Entropy"}:
                self.initialize_shadow_dataset(out_idxs, node_size * n_nets,
                                               self.config.participant["mia_args"]["shadow_model_number"])

        self.indexing_map = net_dataidx_map

        return net_dataidx_map

    def percentage_partition(self, dataset, percentage=0.1):
        """
        Partition a dataset into multiple subsets with a specified level of non-IID-ness.

        This function divides a dataset into several subsets where each subset has a 
        different class distribution, controlled by the 'percentage' parameter. The 
        'percentage' parameter determines the degree of non-IID-ness in the label distribution
        among the federated data subsets.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have 
                                                'data' and 'targets' attributes.
            percentage (float): A value between 0 and 100 that specifies the desired 
                                level of non-IID-ness for the labels of the federated data. 
                                This percentage controls the imbalance in the class distribution 
                                across different subsets.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to number_sub-1) 
                and values are lists of indices corresponding to the samples in each subset.

        The function uses a Dirichlet distribution to create imbalanced proportions of classes
        in each subset. A higher 'percentage' value leads to a more pronounced imbalance in 
        class distribution across subsets. The function ensures that each subset is shuffled 
        and has a unique distribution of classes.

        Example usage:
            federated_data = percentage_partition(my_dataset, 20)
            # This creates federated data subsets with a 20% level of non-IID-ness.
        """
        np.random.seed(self.seed)

        if isinstance(dataset.data, np.ndarray):
            X_train = dataset.data
        elif hasattr(dataset.data, 'numpy'):  # Check if it's a tensor with .numpy() method
            X_train = dataset.data.numpy()
        else:  # If it's a list
            X_train = np.asarray(dataset.data)

        if isinstance(dataset.targets, np.ndarray):
            y_train = dataset.targets
        elif hasattr(dataset.targets, 'numpy'):  # Check if it's a tensor with .numpy() method
            y_train = dataset.targets.numpy()
        else:  # If it's a list
            y_train = np.asarray(dataset.targets)

        num_classes = self.num_classes
        num_subsets = self.number_sub
        class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}

        imbalance_factor = percentage / 100

        subset_indices = [[] for _ in range(num_subsets)]

        for class_idx in range(num_classes):
            indices = class_indices[class_idx]
            np.random.shuffle(indices)
            num_samples_class = len(indices)

            proportions = np.random.dirichlet(np.repeat(1.0 + imbalance_factor, num_subsets))
            proportions = (np.cumsum(proportions) * num_samples_class).astype(int)[:-1]
            split_indices = np.split(indices, proportions)

            for i, subset_idx in enumerate(split_indices):
                subset_indices[i].extend(subset_idx)

        for i in range(num_subsets):
            np.random.shuffle(subset_indices[i])

            class_counts = [0] * num_classes
            for idx in subset_indices[i]:
                _, label = dataset[idx]
                class_counts[label] += 1
            print(f"Partition {i + 1} class distribution: {class_counts}")

        partitioned_datasets = {i: subset_indices[i] for i in range(num_subsets)}

        self.indexing_map = partitioned_datasets

        return partitioned_datasets

    def initialize_shadow_dataset(self, out_idxs, shadow_size, shadow_number):
        """
            Initializes the datasets for training and testing shadow models.

            Args:
                out_idxs (list): List of indices for the out-of-sample training dataset.
                shadow_size (int): Size of each shadow dataset.
                shadow_number (int): Number of shadow datasets to create.

            Raises:
                ValueError: If the remaining training dataset size or the test set size is smaller than the shadow dataset size.

            This method generates random subsets of the training and test datasets to be used as shadow datasets.
            It ensures that the generated shadow datasets are of the specified size and number.
        """
        if len(out_idxs) < shadow_size:
            raise ValueError(
                "The remaining unused training dataset size is even smaller than one shadow training dataset!")
        if len(self.test_set) < shadow_size:
            raise ValueError("The size of test set is even samller than one shadow test dataset!")

        test_indices = np.arange(len(self.test_set))

        shadow_train_indices = []
        shadow_test_indices = []

        for i in range(shadow_number):
            shadow_train_index = np.random.choice(out_idxs, size=shadow_size, replace=True)
            shadow_test_index = np.random.choice(test_indices, size=shadow_size, replace=True)

            shadow_train_indices.append(shadow_train_index)
            shadow_test_indices.append(shadow_test_index)

        print(shadow_train_indices)
        print(shadow_test_indices)
        self.shadow_train = shadow_train_indices
        self.shadow_test = shadow_test_indices

    def initialize_eval_dataset(self, in_idxs, out_idxs):
        """
            Initializes the evaluation datasets.

            Args:
                in_idxs (list): List of indices for the in-sample evaluation dataset.
                out_idxs (list): List of indices for the out-of-sample evaluation dataset.

            This method assigns the provided indices to the class attributes for in-sample and out-sample evaluation datasets.
        """
        self.in_eval = in_idxs
        self.out_eval = out_idxs
        print(self.out_eval)

    def initialize_shadow_dataset1(self, out_idxs, shadow_size, shadow_number):
        """
            Initializes the datasets for training and testing shadow models using a combined dataset approach.

            Args:
                out_idxs (list): List of indices for the out-of-sample training dataset.
                shadow_size (int): Size of each shadow dataset.
                shadow_number (int): Number of shadow datasets to create.

            Returns:
                ConcatDataset: A combined dataset of unused training data and test data.

            Raises:
                ValueError: If the combined size of the remaining training dataset and the test set is smaller than twice the shadow dataset size.

            This method combines the unused training data and the test data into a single dataset, shuffles the combined dataset,
            and selects random indices for the shadow training and testing datasets.
        """
        test_indices = np.arange(len(self.test_set))
        if len(out_idxs) + len(test_indices) < 2 * shadow_size:
            raise ValueError(
                "The remaining unused training dataset size and the test dataset size is smaller than shadow dataset size!")

        unused_train_subset = Subset(self.train_set, out_idxs)
        test_subset = Subset(self.test_set, test_indices)

        combined_dataset = ConcatDataset([unused_train_subset, test_subset])
        total_size = len(combined_dataset)

        np.random.seed(self.seed)
        combined_indices = np.random.permutation(len(combined_dataset))

        shadow_train_indices = combined_indices[:total_size // 2]
        shadow_test_indices = combined_indices[total_size // 2:]

        shadow_train_indices_ls = []
        shadow_test_indices_ls = []

        for i in range(shadow_number):
            shadow_train_index = np.random.choice(shadow_train_indices, size=shadow_size, replace=True)
            shadow_test_index = np.random.choice(shadow_test_indices, size=shadow_size, replace=True)

            shadow_train_indices_ls.append(shadow_train_index)
            shadow_test_indices_ls.append(shadow_test_index)

        self.shadow_train = shadow_train_indices_ls
        self.shadow_test = shadow_test_indices_ls

        return combined_dataset
