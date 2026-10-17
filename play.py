import pygame

from neural_network import brain
from snake_game import snake_game

brain = brain.Brain()
game = snake_game.Main()

state = game.reset()
done = False
clock = pygame.time.Clock()

while not done: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    action = brain.choose_action(state, epsilon=1.0)

    state , reward, done = game.step(action)

    old_state = state
    new_state , reward , done = game.step(action)
    brain.remember(old_state , action , reward , new_state , done)
    brain.learn()

    state = new_state

    game.draw()
    clock.tick(10)

pygame.quit()
print("Game Over! Final Score:", game.score)