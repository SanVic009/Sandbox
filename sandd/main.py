import pygame as pg
import random


pg.init()

# Canvas Dimensions
X = 1000
Y = 900

# Grid vars
MAX = 1000  # its only use is to make the grid
SIZE = 4
ROW = int(Y/SIZE)
COL = int(X/SIZE)


# NOTE: REMOVE THIS VAR
# BOX = 0

# Ball vars
x_ball = int(X/2)
y_ball = int(Y/2)
x_vel = 1
y_vel = 1


# Initialize Canvas
win = pg.display.set_mode((X, Y))
pg.display.set_caption("Sanddd")
clock = pg.time.Clock()

# Initializes the grid matrix with 0 
grid = [[0 for _ in range(COL)] for _ in range(ROW)]

# sand grain falling down
# update the sand bottom up and not top down
def update_sand():
    for row in range(ROW-2, -1, -1):
        for col in range(COL):

            # Checks if the current cell is filled 
            if grid[row][col] == 1  :  
                rand = random.randint(0, 10)
                
                # If the below block is filled or not
                if grid[row+1][col] == 0:
                    grid[row][col] = 0
                    grid[row+1][col] = 1
                
                # Check left diagonal
                elif rand <3 and grid[row+1][(col-1)%COL] == 0:
                    grid[row][col] = 0
                    grid[row+1][(col-1)%COL] = 1
                
                # Check right diagonal
                elif rand >= 7 and grid[row+1][(col+1)%COL] == 0:
                    grid[row][col] = 0
                    grid[row+1][(col+1)%COL] = 1
                

            

def draw_grid():
    for row in range(ROW):
        for col in range(COL):
            if grid[row][col] == 1:
                pg.draw.rect(win, (255, 255, 255), (col*SIZE, row*SIZE , SIZE, SIZE))


# grid[2][1] = 1

if __name__ == "__main__":

    # Game Loop
    run = True
    while run:
        win.fill((0,0,0))
        draw_grid()

        radius = 5
        pg.draw.circle(win, (212, 42, 161), (x_ball, y_ball), radius)

        # Logic for quitting
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        # Draw the grid
        # for x in range(1, MAX+1, SIZE):
        #     pg.draw.line(win, (255, 255, 255), (x, 0), (x, MAX) )
        #     pg.draw.line(win, (255, 255, 255), (0, x), (MAX, x) )

        user_input = pg.key.get_pressed()
        if user_input[pg.K_SPACE]:
            x_index = int(x_ball/SIZE)
            y_index = int(y_ball/SIZE)
            grid[y_index][x_index] = 1
        
        update_sand()
        # TODO: - make the sand fall on the left and the right
        #       - make sure that it first falls flat before turningg right or left. First check if there  is  a sand particle below the current particle or not. Then only go to the left or the right

        if user_input[pg.K_LEFT] :
            x_ball = (x_ball - x_vel)%X
        if user_input[pg.K_RIGHT] :
            x_ball = (x_ball + x_vel)%X
        if user_input[pg.K_UP] and y_ball > 0:
            y_ball -= y_vel
        if user_input[pg.K_DOWN] and y_ball < Y:
            y_ball += y_vel
        if user_input[pg.K_q]:
            run = False
            print("Byee")
        
        pg.display.update()
