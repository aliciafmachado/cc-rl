from cc_rl.gym_cc.Renderer import Renderer
import random

DEPTH = 10
ITERATIONS = 20

tree = {}  # Map action -> prob, tree
cur_branch = tree
renderer = Renderer('draw', DEPTH+1)

for i in range(ITERATIONS):
    for j in range(DEPTH):
        action = random.randint(0, 1) * 2 - 1
        if action in cur_branch:
            probability = cur_branch[action][0]
            cur_branch = cur_branch[action][1]
        else:
            if -action in cur_branch:
                probability = 1 - cur_branch[-action][0]
            else:
                probability = random.random()
            cur_branch[action] = probability, {}
            cur_branch = cur_branch[action][1]
        
        renderer.render(action, probability)

    renderer.reset()
    cur_branch = tree
