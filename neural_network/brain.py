import torch    
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11 , 32),
            nn.ReLU(),

            nn.Linear(32, 64),  
            nn.ReLU(),
            
            nn.Linear(64, 32),  
            nn.ReLU(),

            nn.Linear(32, 3)
        )
        self.memory = deque(maxlen=10_000)

        self.gamma = 0.9
        self.batch_size = 64
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995


    def forward(self, state):
        return self.layers(state)

    def choose_action(self , state , epsilon = 1.0):
        if random.random() < self.epsilon:
            return random.randrange(3)  # Random action

        state_tensor = torch.tensor(state , dtype=torch.float32)
        with torch.no_grad():
            scores = self(state_tensor)

        return int(scores.argmax().item()) 

    def remember(self , state , action , reward , next_state , done):
        experience = (state , action , reward , next_state , done)
        self.memory.append(experience)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory , self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states , dtype=torch.float32)
        actions_tensor = torch.tensor(actions , dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards , dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states , dtype=torch.float32)
        dones_tensor = torch.tensor(dones , dtype=torch.float32).unsqueeze(1)

        current_q_values = self(states_tensor).gather(1 , actions_tensor)
        next_q_values = self(next_states_tensor).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))

        loss = self.loss_fn(current_q_values , target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    brain = Brain()
    state = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    action = brain.choose_action(state , epsilon=0.0)

    print(action)