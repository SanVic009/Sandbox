import pygame as pg
import random


pg.init()

# Canvas Dimensions
X = 1000
Y = 900

# Grid vars
MAX = 1000
SIZE = 5
BOX = int(MAX/SIZE)

# Ball vars
x_ball = int(X/2)
y_ball = int(Y/2)
x_vel = 1
y_vel = 1

run = True
radius = 5

# Initialize Canvas
win = pg.display.set_mode((X, Y))
pg.display.set_caption("Sanddd")
clock = pg.time.Clock()


# Initializes the grid matrix with 0 
grid = [[0 for _ in range(BOX)] for _ in range(BOX)]

# sand grain  falling down
def update_sand():
    for row in range(BOX-1):
        for col in range(BOX-1):
            rand = random.randint(0, 10)
            if grid[row][col] == 1 and grid[row+1][col] == 0:
                grid[row][col] = 0
                grid[row+1][col] = 1

            if grid[row+1][col]==1 and rand >= 5 and grid[row+1][col-1] == 0 :  #Take the grain to right 
                grid[row][col] = 0
                grid[row+1][col+1] = 1
            
            elif grid[row+1][col]==1 and rand < 5 and grid[row+1][col+1] == 0:   #Take the grain to left
                grid[row][col] = 0
                grid[row+1][col-1] = 1
            
    

def draw_grid():
    for row in range(BOX):
        for col in range(BOX):
            if grid[row][col] == 1:
                pg.draw.rect(win, (255, 255, 255), (col*SIZE, row*SIZE , SIZE, SIZE))

grid[2][1] = 1

if __name__ == "__main__":
    # Game Loop
    while run:
        win.fill((0,0,0))
        draw_grid()
        pg.draw.circle(win, (212, 42, 161), (x_ball, y_ball), radius)

        # Logic for quitting
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        # # Draw the grid
        # for x in range(0, MAX+1, SIZE):
        #     pg.draw.line(win, (255, 255, 255), (x, 0), (x, MAX) )
        #     pg.draw.line(win, (255, 255, 255), (0, x), (MAX, x) )

        user_input = pg.key.get_pressed()
        if user_input[pg.K_SPACE]:
            x_index = int(x_ball/SIZE)
            y_index = int(y_ball/SIZE)
            grid[y_index][x_index] = 1
        
        update_sand()
        













        if user_input[pg.K_LEFT] and x_ball > 0:
            x_ball -= x_vel
        if user_input[pg.K_RIGHT] and x_ball < X:
            x_ball += x_vel
        if user_input[pg.K_UP] and y_ball > 0:
            y_ball -= y_vel
        if user_input[pg.K_DOWN] and y_ball < Y:
            y_ball += y_vel
        if user_input[pg.K_q]:
            run = False
            print("Byee")
        
        pg.display.update()
