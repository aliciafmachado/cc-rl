from src.gym_cc.Renderer import Renderer

path = [-1, 1, 1, -1, 1, 1, -1, -1, 1, -1]
probabilities = [0.5, 0.4, 0.3, 0.2, 0.1, 0.4, 0.5, 0.2, 0.1]
renderer = Renderer('draw', path, probabilities)

for i in range(len(path)):
    renderer.render(i)
renderer.reset()
