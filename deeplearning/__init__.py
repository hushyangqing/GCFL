from .dataset import UserDataset, assign_user_data
from .networks import NaiveMLP, NaiveCNN

nn_registry = {
"naiveMLP": NaiveMLP,
"naiveCNN": NaiveCNN,
}
