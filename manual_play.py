import pygame
import sys
from rubarrel import Rubarrel

class Barrel:
    def __init__(self):
        self.direction = "Up"

    def turn(self):
        if self.direction == "Up":
            self.direction = "Right"
        elif self.direction == "Right":
            self.direction = "Down"
        elif self.direction == "Down":
            self.direction = "Left"
        elif self.direction == "Left":
            self.direction = "Up"

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Type and Display")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.monospace_font = pygame.font.SysFont("monospace", 36)
        self.text = ""
        self.barrel = Rubarrel()
        self.number_colors = {
            '0': (255, 0, 0),  # Red
            '1': (0, 255, 0),  # Green
            '2': (0, 0, 255),  # Blue
            '3': (155, 155, 0),  # Yellow
            '4': (255, 0, 255),  # Magenta
            '5': (0, 255, 255)}  # Cyan
    def run(self):
        running = True
        while running:
            self.screen.fill((255, 255, 255))
            self.render_text()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.text = ""
                    elif event.key == pygame.K_e:
                        self.barrel.turn("left", 1)
                        self.text = self.barrel.__repr__()
                    elif event.key == pygame.K_d:
                        self.barrel.turn("left", -1)
                        self.text = self.barrel.__repr__()
                    elif event.key == pygame.K_r:
                        self.barrel.turn("right", 1)
                        self.text = self.barrel.__repr__()
                    elif event.key == pygame.K_f:
                        self.barrel.turn("right", -1)
                        self.text = self.barrel.__repr__()  # Set text to barrel direction after spacebar press
                    elif event.key == pygame.K_SPACE:
                        self.barrel.shift()
                        self.text = self.barrel.__repr__()  # Set text to barrel direction after spacebar press

    def render_text(self):
        lines = self.text.split('\n')
        y_offset = 0
        for line in lines:
            x_offset = 0
            for char in line:
                if char.isdigit():  # Check if character is a number
                    text_surface = self.monospace_font.render(char, True, self.number_colors[char])  # Red color for numbers
                else:
                    text_surface = self.monospace_font.render(char, True, (0, 0, 0))  # Black color for other characters
                text_rect = text_surface.get_rect(topleft=( x_offset, self.screen_height // 2 + y_offset))
                self.screen.blit(text_surface, text_rect)
                x_offset += text_surface.get_width() + 10  # Adjust spacing between characters
            y_offset += self.monospace_font.get_height() + 10  # Adjust spacing between lines


if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()
