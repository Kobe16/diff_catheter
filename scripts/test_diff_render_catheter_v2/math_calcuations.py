import torch

p_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866])
p_start = torch.tensor([0.02, 0.002, 0.0])

# print("p_start - 0.01: ", p_start - 0.01)
# print("p_init - 0.01: ", p_init - 0.01)

# print("p_start[2] + 0.05: ", p_start[2] + 0.05)
# print("p_init[2] + 0.05: ", p_init[2] + 0.05)
# print("p_init[5] + 0.05: ", p_init[5] + 0.05)

print("p_start[2] + 0.1: ", p_start[2] + 0.1)
print("p_init[2] + 0.1: ", p_init[2] + 0.1)
print("p_init[5] + 0.1: ", p_init[5] + 0.1)