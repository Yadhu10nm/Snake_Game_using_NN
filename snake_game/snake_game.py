import pygame
import random

pygame.init()

WIDTH = 900
HEIGHT = 600
CELL = 30
display = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Direction order: right, down, left, up
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class Snake:
    def __init__(self):
        self.x = 150
        self.y = 150
        self.body = [pygame.Rect(self.x, self.y, CELL, CELL)]
        self.direction = 0  # starts moving right

    def move(self):
        dx, dy = DIRECTIONS[self.direction]
        self.x += dx * CELL
        self.y += dy * CELL
        self.body.insert(0, pygame.Rect(self.x, self.y, CELL, CELL))

    def draw_snake(self):
        for i, block in enumerate(self.body):
            color = "red" if i == 0 else "green"
            pygame.draw.rect(display, color, block)

    def set_direction(self, direction):
        # Prevent a 180-degree turn.
        if direction != (self.direction + 2) % 4:
            self.direction = direction


class Fruit:
    def __init__(self):
        self.body = None
        self.get_random_pos([])

    def get_random_pos(self, snake_body):
        occupied = {(block.x, block.y) for block in snake_body}
        free_cells = [
            (x, y)
            for x in range(0, WIDTH, CELL)
            for y in range(0, HEIGHT, CELL)
            if (x, y) not in occupied
        ]
        if free_cells:
            x, y = random.choice(free_cells)
            self.body = pygame.Rect(x, y, CELL, CELL)

    def draw_fruit(self):
        pygame.draw.rect(display, "orange", self.body)


def draw_grids():
    for x in range(0, WIDTH, CELL):
        for y in range(0, HEIGHT, CELL):
            pygame.draw.rect(display, (40, 40, 40), (x, y, CELL, CELL), 1)


class Main:
    def __init__(self):
        self.snake = Snake()
        self.fruit = Fruit()
        self.fruit.get_random_pos(self.snake.body)
        self.score = 0
        self.done = False
        self.font = pygame.font.SysFont(None, 40)

    def reset(self):
        self.snake = Snake()
        self.fruit = Fruit()
        self.fruit.get_random_pos(self.snake.body)
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        head = self.snake.body[0]
        x , y = head.x , head.y
        direction = self.snake.direction

        straight = direction
        right = (direction + 1) % 4
        left = (direction - 1) % 4

        def danger(direction_index):
            dx , dy = DIRECTIONS[direction_index]
            next_x = x + dx * CELL
            next_y = y + dy * CELL
            next_rect = pygame.Rect(next_x, next_y, CELL, CELL)
            hit_wall = (
                next_x < 0 or next_x >= WIDTH
                or
                next_y < 0 or next_y >= HEIGHT
            )
            hits_body = any(next_rect.colliderect(part) for part in self.snake.body[:-1])
            return int(hit_wall or hits_body)
        
        fruit_x, fruit_y = self.fruit.body.x, self.fruit.body.y

        return [
            danger(straight),
            danger(right),
            danger(left),
            int(direction == 2),  # moving left
            int(direction == 0),  # moving right
            int(direction == 3),  # moving up
            int(direction == 1),  # moving down
            int(fruit_x < x),     # fruit left
            int(fruit_x > x),     # fruit right
            int(fruit_y < y),     # fruit up
            int(fruit_y > y),     # fruit down
        ]
    

    def step(self, action):
        """
        Advance the game one cell.
        action: 0 = straight, 1 = turn right, 2 = turn left
        Returns: state, reward, done
        """
        if self.done:
            return self.get_state(), 0, True

        if action == 1:
            self.snake.direction = (self.snake.direction + 1) % 4
        elif action == 2:
            self.snake.direction = (self.snake.direction - 1) % 4

        self.snake.move()
        head = self.snake.body[0]
        eating = head.colliderect(self.fruit.body)

        # If not eating, the tail moves away, so exclude it from collision.
        body_to_check = self.snake.body if eating else self.snake.body[:-1]
        hit_body = any(head.colliderect(block) for block in body_to_check[1:])

        hit_wall = (
            head.left < 0 or head.top < 0
            or head.right > WIDTH or head.bottom > HEIGHT
        )

        if hit_wall or hit_body:
            self.done = True
            return self.get_state(), -10, True

        if eating:
            self.score += 1
            self.fruit.get_random_pos(self.snake.body)
            reward = 10
        else:
            self.snake.body.pop()
            reward = -0.1

        return self.get_state(), reward, False

    def draw(self):
        display.fill("black")
        draw_grids()
        self.snake.draw_snake()
        self.fruit.draw_fruit()

        score_surface = self.font.render(f"Score: {self.score}", True, "white")
        display.blit(score_surface, (10, 10))
        pygame.display.flip()

    

if __name__ == "__main__":
    game = Main() 
