import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 1200
RESOLU = 0.02
MAP_WIDTH = 30
MAP_HEIGHT = 30
ORIGIN = np.array([-MAP_WIDTH/2,-MAP_HEIGHT/2])
GRID_WIDTH = int(MAP_WIDTH/RESOLU)
GRID_HEIGHT = int(MAP_HEIGHT/RESOLU)
GRID_ORIGIN_OFFSET = np.array([GRID_WIDTH//2,GRID_HEIGHT//2])

num_lines = 1024

def cvdraw(map, start, end):
    map = cv2.line(
        map,
        (np.array(start-ORIGIN)/RESOLU).astype(np.int_),
        (np.array(end-ORIGIN)/RESOLU).astype(np.int_),
        255,)
    return map

def parallel_bresenham(map, start, end):
    end = ((end-ORIGIN)/RESOLU).astype(np.int_)
    start = ((start-ORIGIN)/RESOLU).astype(np.int_)

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    invert_y = False
    invert_x = False
    swap_xy = False

    if dy <= 0:
        dy = -dy
        invert_y = True

    if dx <= 0:
        dx = -dx
        invert_x = True

    if dy >= dx:
        dx, dy = dy, dx
        swap_xy = True

    k = (dy+1)/(dx+1)

    for i in range(dx+1):
        grid_x = i
        # grid_y = i# - grid_x
        grid_y = int(grid_x*k+0.49*k)
        if swap_xy:
            grid_x, grid_y = grid_y, grid_x
        if invert_y:
            grid_y = -grid_y
        if invert_x:
            grid_x = -grid_x
        if grid_y+start[1] < GRID_HEIGHT and grid_x+start[0] < GRID_WIDTH and grid_y+start[1] >= 0 and grid_x+start[0] >= 0:
            map[grid_y+start[1],grid_x+start[0]] = 255

    return map

def drawMap(name, map):
    map = cv2.resize(map, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST) # type: ignore
    cv2.imshow(name, map)

def drawHAHAHA(map, start, ft):
    # map2 = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

    lidar_resolu = math.pi*2/num_lines

    for i in range(num_lines):
        end = start + np.array([math.cos(i*lidar_resolu),math.sin(i*lidar_resolu)])*ft[i]
        map = cvdraw(map, start, end)
        # map2 = parallel_bresenham(map, start, end)
    drawMap("cv", map)

def main():
    lidar_range = 5
    # ft = 2.5
    ft = np.zeros(num_lines)
    ft[0] = 2.5
    for i in range(1, num_lines):
        ft[i] = 0.7*ft[i-1] + 0.3*(random.uniform(0,1)*lidar_range)

    map = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)

    for(i) in range(360):
        drawHAHAHA(map, start=np.array([math.cos(i*0.1),math.sin(i*0.1)])*3, ft=ft)
        cv2.waitKey(80)


if __name__ == "__main__":
    main()