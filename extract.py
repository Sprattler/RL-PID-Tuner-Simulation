import torch
from Agent_Pendel import Actor  # oder wie deine Actor-Klasse heißt

state_dim = 2
action_dim = 1

actor = Actor(state_dim, action_dim)
actor.load_state_dict(torch.load("td3_actor.pth"))
actor.eval()

weights = actor.pi_layer.weights.detach().cpu().numpy()
Kp, Ki = abs(weights[0])  # abs() weil im Forward abs() verwendet wird

print(f"Kp = {Kp:.4f}")
print(f"Ki = {Ki:.4f}")
