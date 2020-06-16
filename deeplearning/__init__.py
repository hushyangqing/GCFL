from .dataset import usersOwnData, userDataset
from .networks import naiveMLP, naiveCNN

NNRegistry = {
"naiveMLP": naiveMLP,
"naiveCNN": naiveCNN,
}
