import gym
from gym import spaces
import random
import numpy as np
from math import sin, cos
from utils import ray, forward, collision, distance
import map_w
import pygame

class Player:
    def __init__(self, x, y, num):
        self.x = x
        self.y = y
        if num == 1:
            self.rot = 270
        else:
            self.rot = 90
        self.num = num
        self.ammoT = 90
        self.ammo = 30
        self.speed = 1
        self.seen_player = False
        self.seenx = 0
        self.seeny = 0

    def up(self):
        if self.y - self.speed > -1:
            for wall in map_w.walls:
                if collision(self.x, self.y - self.speed, wall[0], wall[1], 1, 1, 1, 1):
                    break
            else:
                self.y -= self.speed

    def down(self):
        if self.y + self.speed < len(map_w.MAP):
            for wall in map_w.walls:
                if collision(self.x, self.y + self.speed, wall[0], wall[1], 1, 1, 1, 1):
                    break
            else:
                self.y += self.speed

    def right(self):
        if self.x + self.speed < len(map_w.MAP[0]):
            for wall in map_w.walls:
                if collision(self.x + self.speed, self.y, wall[0], wall[1], 1, 1, 1, 1):
                    break
            else:
                self.x += self.speed
    
    def left(self):
        if self.x - self.speed > -1:
            for wall in map_w.walls:
                if collision(self.x - self.speed, self.y, wall[0], wall[1], 1, 1, 1, 1):
                    break
            else:
                self.x -= self.speed

class AirsoftEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    p1x = 9
    p1y = 11
    p2x = 1
    p2y = 1
    DISTANCE_RADIUS = 0.5
    MAX_ITER = 5000

    def __init__(self, start_model = None):
        self.selfplay = start_model
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(99,))
        self.players = [Player(self.p1x, self.p1y, 1), Player(self.p2x, self.p2y, -1)]
        self.IS_PLAYER1 = random.random() <= 0.5
        self.iteration = 0
        self.rays = []
        self.sr = []

        self.screen = None
        self.clock = None

    def reset(self):
        self.sr = []
        self.players = [Player(self.p1x, self.p1y, 1), Player(self.p2x, self.p2y, -1)]
        self.IS_PLAYER1 = random.random() <= 0.5
        self.iteration = 0
        self.rays = []
        
        if self.IS_PLAYER1:
            return self.get_obs(self.players[0], self.players[1])
        else:
            return self.get_obs(self.players[1], self.players[0])

    def get_obs(self, player: Player, opponent: Player):
        self.rays = []
        obs = []
        for angle in range(-45, 45):
            self.rays.append(ray(player.x, player.y, player.rot + angle * 0.01, map_w.walls, opponent.x, opponent.y))
        for i in range(len(self.rays)):
            obs.append(self.rays[i][0])
            if not player.seen_player:
                player.seen_player = self.rays[i][1]
            if player.seen_player and self.rays[i][1]:
                player.seenx = opponent.x
                player.seeny = opponent.y
        obs.append(player.x)
        obs.append(player.y)
        obs.append(player.rot % 360)
        obs.append(1 if player.seen_player else 0)
        obs.append(player.seenx)
        obs.append(player.seeny)
        obs.append(player.ammo)
        obs.append(player.ammoT)
        obs.append(np.degrees(np.arctan2(player.seeny - player.y, player.seenx - player.x)))

        return obs

    def reload(self, player: Player):
        how_much = 30 - player.ammo
        if player.ammoT >= how_much:
            player.ammo += how_much
            player.ammoT -= how_much

    def get_action(self, action, p: Player, o: Player, acPlayer = True):
        done = False
        reward = 0

        p.rot = action[8]

        move = np.argmax(action[0:5])

        if move == 0:
            p.up()
        elif move == 1:
            p.down()
        elif move == 2:
            p.right()
        else:
            p.left()

        move = np.argmax(action[5:8])

        if move == 0:
            if p.ammo > 0:
                shoot_ray = ray(p.x, p.y, p.rot, map_w.walls, o.x, o.y)
                if shoot_ray[1]:
                    self.sr.append([p.x, p.y, o.x, o.y])
                else:
                    self.sr.append([p.x, p.y, shoot_ray[2], shoot_ray[3]])
                if distance(shoot_ray[2], shoot_ray[3], o.x, o.y) <= self.DISTANCE_RADIUS and acPlayer:
                    reward += 0.1
                if shoot_ray[1]:
                    done = True
                    if acPlayer:
                        reward += 100
                    else:
                        reward -= 100
                p.ammo -= 1
        elif move == 1:
            self.reload(p)

        if not done and p.ammo + p.ammoT == 0:
            reward -= 10

        return reward, done

    def step(self, action):
        done = self.iteration >= self.MAX_ITER
        reward = 0
        self.sr = []
        
        if self.IS_PLAYER1:
            p = self.players[0]
            o = self.players[1]
        else:
            p = self.players[1]
            o = self.players[0]

        if not done:
            reward, done = self.get_action(action, p, o, True)
            if not done:
                if self.selfplay is None:
                    reward, done = self.get_action(self.action_space.sample(), o, p, False)
                else:
                    reward, done = self.get_action(self.selfplay.predict(self.get_obs(o, p))[0], o, p, False)

        self.iteration += 1

        return self.get_obs(p, o), reward, done, {}

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((500, 500), pygame.RESIZABLE)
            pygame.display.set_caption("Airsoft Game")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))

        x, y = self.screen.get_size()

        w = x/len(map_w.MAP[0])
        h = y/len(map_w.MAP)

        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.players[0].x * w, self.players[0].y * h, 1 * w, 1 * h))
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.players[1].x * w, self.players[1].y * h, 1 * w, 1 * h))
        pygame.draw.circle(self.screen, (0, 255, 255), (self.players[0].x * w, self.players[0].y * h), 5)
        pygame.draw.circle(self.screen, (0, 255, 255), (self.players[1].x * w, self.players[1].y * h), 5)

        for i in range(len(map_w.walls)):
            pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(map_w.walls[i][0] * w, map_w.walls[i][1] * h, 1 * w, 1 * h))

        for i in range(len(self.sr)):
            pygame.draw.line(self.screen, (255, 0, 255), (self.sr[i][0] * w, self.sr[i][1] * h), (self.sr[i][2] * w, self.sr[i][3] * h))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(15)
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return True