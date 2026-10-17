import pygame
import torch
import torch.nn as nn


GAME_WIDTH = 900


class NetworkVisualizer:
    def __init__(self, brain):
        self.brain = brain

        screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
        visualizer_width = max(400, screen_width - GAME_WIDTH)

        self.window = pygame.Window(
            "Snake Neural Network",
            size=(visualizer_width, screen_height),
            position=(GAME_WIDTH, 0),
            resizable=True,
        )

        self.surface = self.window.get_surface()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 16)
        self.closed = False

        # This visualizer expects Brain.layers to be an nn.Sequential
        # containing Linear and ReLU layers.
        self.linear_layers = [
            layer
            for layer in self.brain.layers
            if isinstance(layer, nn.Linear)
        ]

    def handle_event(self, event):
        if (
            event.type == pygame.WINDOWCLOSE
            and event.window.id == self.window.id
        ):
            self.window.destroy()
            self.closed = True

    def update(self, state, action, reward, epsilon, episode, score):
        if self.closed:
            return

        self.surface = self.window.get_surface()
        width, height = self.surface.get_size()
        self.surface.fill((16, 24, 32))

        # Run the real network and save the activation after each hidden ReLU.
        device = next(self.brain.parameters()).device
        values = torch.as_tensor(state, dtype=torch.float32, device=device)
        hidden_activations = []

        with torch.no_grad():
            for layer in self.brain.layers:
                values = layer(values)

                if isinstance(layer, nn.ReLU):
                    hidden_activations.append(values.detach().cpu())

            outputs = values.detach().cpu()

        # Values for each displayed column: input, hidden layers, output.
        activation_columns = [
            torch.as_tensor(state, dtype=torch.float32)
        ] + hidden_activations + [outputs]

        layer_values = [column.tolist() for column in activation_columns]
        layer_sizes = [len(column) for column in layer_values]
        layer_count = len(layer_values)

        # Spread all columns evenly across the resizable window.
        left_margin = width * 0.08
        right_margin = width * 0.08
        if layer_count == 1:
            x_positions = [width / 2]
        else:
            gap = (width - left_margin - right_margin) / (layer_count - 1)
            x_positions = [
                left_margin + i * gap for i in range(layer_count)
            ]

        y_positions = [
            self._positions(size, height * 0.12, height * 0.90)
            for size in layer_sizes
        ]

        # Header and layer labels.
        self._text(
            f"Episode {episode}   Score {score}   "
            f"Reward {reward:+.1f}   Epsilon {epsilon:.3f}",
            width // 2,
            24,
            centered=True,
        )

        self._text(
            f"INPUT ({layer_sizes[0]})",
            x_positions[0],
            58,
            centered=True,
        )

        for i, size in enumerate(layer_sizes[1:-1], start=1):
            self._text(
                f"HIDDEN {i} ({size})",
                x_positions[i],
                58,
                centered=True,
                small=True,
            )

        self._text(
            f"ACTIONS ({layer_sizes[-1]})",
            x_positions[-1],
            height * 0.34,
            centered=True,
        )

        # Draw the weights between every pair of adjacent columns.
        # Each Linear layer's weight shape is [next_neurons, current_neurons].
        for layer_index, linear in enumerate(self.linear_layers):
            weights = linear.weight.detach().cpu()

            for next_i in range(weights.shape[0]):
                for current_i in range(weights.shape[1]):
                    self._connection(
                        x_positions[layer_index],
                        y_positions[layer_index][current_i],
                        x_positions[layer_index + 1],
                        y_positions[layer_index + 1][next_i],
                        float(weights[next_i, current_i]),
                    )

        # Draw neurons after the connections so they appear on top.
        for column_i, activations in enumerate(layer_values):
            max_activation = max(
                (abs(float(value)) for value in activations),
                default=1.0,
            ) or 1.0

            for neuron_i, value in enumerate(activations):
                normalized_value = abs(float(value)) / max_activation

                selected = (
                    column_i == layer_count - 1
                    and neuron_i == action
                )

                radius = self._neuron_radius(layer_sizes[column_i])

                self._neuron(
                    x_positions[column_i],
                    y_positions[column_i][neuron_i],
                    normalized_value,
                    radius=radius,
                    selected=selected,
                )

        # Label the output scores.
        action_names = ["Straight", "Right", "Left"]
        output_values = layer_values[-1]

        for i, value in enumerate(output_values):
            name = action_names[i] if i < len(action_names) else f"Action {i}"
            self._text(
                f"{name}: {float(value):.2f}",
                x_positions[-1],
                y_positions[-1][i] + 20,
                centered=True,
                small=True,
            )

        self.window.flip()

    @staticmethod
    def _positions(count, top, bottom):
        if count <= 0:
            return []

        if count == 1:
            return [(top + bottom) / 2]

        gap = (bottom - top) / (count - 1)
        return [top + i * gap for i in range(count)]

    @staticmethod
    def _neuron_radius(layer_size):
        if layer_size >= 100:
            return 3
        if layer_size >= 50:
            return 5
        return 8

    def _connection(self, x1, y1, x2, y2, weight):
        # Cyan represents a positive weight; pink represents a negative one.
        color = (45, 195, 255) if weight >= 0 else (255, 75, 110)
        thickness = max(1, min(3, int(abs(weight) * 5)))

        pygame.draw.line(
            self.surface,
            color,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            thickness,
        )

    def _neuron(self, x, y, activation, radius, selected=False):
        strength = max(0.0, min(1.0, float(activation)))
        green = int(65 + strength * 190)

        fill_color = (30, green, 110)
        outline_color = (
            (255, 209, 102) if selected else (230, 240, 245)
        )
        outline_width = 3 if selected else 1

        pygame.draw.circle(
            self.surface,
            fill_color,
            (int(x), int(y)),
            radius,
        )
        pygame.draw.circle(
            self.surface,
            outline_color,
            (int(x), int(y)),
            radius,
            outline_width,
        )

    def _text(self, value, x, y, centered=False, small=False):
        font = self.small_font if small else self.font
        image = font.render(str(value), True, (240, 245, 250))
        rect = image.get_rect()

        if centered:
            rect.center = (int(x), int(y))
        else:
            rect.topleft = (int(x), int(y))

        self.surface.blit(image, rect)