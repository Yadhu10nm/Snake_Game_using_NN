import os
os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"

import pygame
import time
from pathlib import Path

import torch

from neural_network.brain import Brain
from snake_game.snake_game import Main
from vizualizer.network import NetworkVisualizer


brain = Brain()
game = Main()
visualizer = NetworkVisualizer(brain)
clock = pygame.time.Clock()

start = time.time()

episodes = 2000
for episode in range(episodes + 1):
    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.WINDOWCLOSE:
                if event.window.id == visualizer.window.id:
                    visualizer.handle_event(event)
                else:
                    pygame.quit()
                    raise SystemExit

        action = brain.choose_action(state)
        next_state, reward, done = game.step(action)

        brain.remember(state, action, reward, next_state, done)
        brain.learn()

        state = next_state
        visualizer.update(
            state, 
            action, 
            reward, 
            brain.epsilon, 
            episode + 1, 
            game.score)
        
        game.draw()
        clock.tick(100)

    brain.epsilon = max(
    brain.epsilon_min,
    brain.epsilon * brain.epsilon_decay,
    )
    print(f"Episode {episode + 1} / {episodes}:\t\tscore: {game.score}\tepsilon: {brain.epsilon:.3f}")
    if episode == 500 or episode == 1000 or episode == 1500 or episode == 2000:
        model_path = Path(__file__).resolve().parent / f"snake_brain_{episode}.pth"
        torch.save(brain.state_dict(), model_path)
        print(f"Model({episode}) saved to {model_path}")

model_path = Path(__file__).resolve().parent / "snake_brain.pth"
torch.save(brain.state_dict(), model_path)
print(f"Model saved to {model_path}")

print(f"Training completed in {time.time() - start:.2f} seconds.")
pygame.quit()