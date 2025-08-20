# rnn_visualizer.py
import numpy as np

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
from rnn_math import run_rnn_sequence  # <-- your existing math module


# ==== Window & Grid ====
W, H = 1400, 700                 # window size
GRID_R, GRID_C = 5, 5            # grid rows, cols
CELL_W, CELL_H = 120, 120        # size of each grid cell
H_GAP = 60                       # horizontal gap between workspaces

# ==== Box Sizes & Text ====
BOX_W_SMALL, BOX_H_SMALL = 80, 60    # for vectors & scalars
BOX_W_LARGE, BOX_H_LARGE = 120, 80   # for 2×2 weight matrices
LINE_HEIGHT = 14                     # pixels between text lines

# Default box width and height for arrows to use
BOX_W = BOX_W_LARGE  # or BOX_W_SMALL, depending on your preference
BOX_H = BOX_H_LARGE  # or BOX_H_SMALL


# ==== Example Inputs & History ====
inputs = [np.array([1,0]), np.array([0,1]), np.array([1,0])]
history = run_rnn_sequence(inputs)   # precompute all timesteps
time_step = 0                        # how many to display

scroll_x = 0  # horizontal scroll offset

# ==== Drawing Helpers ====
def draw_text(x, y, s):
    glRasterPos2f(x, y)
    for ch in s:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))

def draw_box(cx, cy, w, h, label, val=""):
    x0, y0 = cx - w/2, cy - h/2
    # box border
    glColor3f(0,0,0)
    glBegin(GL_LINE_LOOP)
    for dx,dy in [(0,0),(w,0),(w,h),(0,h)]:
        glVertex2f(x0+dx, y0+dy)
    glEnd()
    # label (top-left)
    draw_text(x0+5, y0 + h - LINE_HEIGHT, label)
    # value (multiline)
    if val:
        lines = val.split("\n")
        for i, line in enumerate(lines):
            draw_text(x0+5, y0 + h - LINE_HEIGHT*(2+i), line)

def draw_arrow(x1,y1, x2,y2, lab=""):
    glColor3f(0,0,0)
    glBegin(GL_LINES)
    glVertex2f(x1,y1); glVertex2f(x2,y2)
    glEnd()
    ang = math.atan2(y2-y1, x2-x1); sz=6
    glBegin(GL_TRIANGLES)
    glVertex2f(x2,y2)
    glVertex2f(x2 - sz*math.cos(ang-0.3), y2 - sz*math.sin(ang-0.3))
    glVertex2f(x2 - sz*math.cos(ang+0.3), y2 - sz*math.sin(ang+0.3))
    glEnd()
    if lab:
        draw_text((x1+x2)/2, (y1+y2)/2 + 8, lab)

def grid_cell_center(r, c, ox, oy):
    x = ox + c*CELL_W + CELL_W/2
    y = oy - r*CELL_H - CELL_H/2
    return x, y

def format_vector(v):
    return "\n".join(f"{val:.2f}" for val in v)

def format_matrix(m):
    return "\n".join(" ".join(f"{val:.2f}" for val in row) for row in m)

def link(fr, fc, tr, tc, ox, oy, box_w, box_h, lab=""):
    x1, y1 = grid_cell_center(fr, fc, ox, oy)
    x2, y2 = grid_cell_center(tr, tc, ox, oy)

    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return

    unit_dx, unit_dy = dx / length, dy / length
    x1b = x1 + unit_dx * box_w / 2
    y1b = y1 + unit_dy * box_h / 2
    x2b = x2 - unit_dx * box_w / 2
    y2b = y2 - unit_dy * box_h / 2

    draw_arrow(x1b, y1b, x2b, y2b, lab)

# ==== Draw one timestep workspace ====

def draw_workspace(ox, oy, data):
    t = data['t']
    # positions in the 5×5 grid
    pos = {
      "y":      (0,2),
      "Wya":    (1,2),
      "a_prev": (2,0),
      "Waa":    (2,1),
      "RNN":    (2,2),
      "a":      (2,3),
      "Wax":    (3,2),
      "x":      (4,2),
    }
    # prepare string values
    vals = {
      "x":      format_vector(data['x']),
      "a_prev": format_vector(data['a_prev']),
      "a":      format_vector(data['a']),
      "y":      format_vector(data['y']),
      "Wax":    format_matrix(data['Wax']),
      "Waa":    format_matrix(data['Waa']),
      "Wya":    format_matrix(data['Wya']),
    }
    # draw boxes
    for label, (r,c) in pos.items():
        cx, cy = grid_cell_center(r,c,ox,oy)
        # pretty label
        if label=="a_prev": txt = f"a<{t-1}>"
        elif label=="a":    txt = f"a<{t}>"
        else:               txt = f"{label}<{t}>"
        # pick size
        if label=="RNN":
            w,h = BOX_W_SMALL, BOX_H_SMALL
            v = ""
        elif label in ("Wax","Waa","Wya"):
            w,h = BOX_W_LARGE, BOX_H_LARGE
            v = vals[label]
        else:
            w,h = BOX_W_SMALL, BOX_H_SMALL
            v = vals[label]
        draw_box(cx, cy, w, h, txt, v)

    # draw arrows
    



    
    link(4, 2, 3, 2, ox, oy, BOX_W, BOX_H, "Wax")
    link(3, 2, 2, 2, ox, oy, BOX_W, BOX_H)
    link(2, 0, 2, 1, ox, oy, BOX_W, BOX_H)
    link(2, 1, 2, 2, ox, oy, BOX_W, BOX_H, "Waa")
    link(2, 2, 2, 3, ox, oy, BOX_W, BOX_H)
    link(2, 2, 1, 2, ox, oy, BOX_W, BOX_H)
    link(1, 2, 0, 2, ox, oy, BOX_W, BOX_H, "Wya")


# ==== GLUT Callbacks ====
def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    if time_step == 0:
        draw_text(20,20, "Press SPACE to step through RNN")
    for i in range(time_step):
        ox = 50 + i*(GRID_C*CELL_W + H_GAP) + scroll_x
        oy = H - 50
        draw_workspace(ox, oy, history[i])
    glutSwapBuffers()

def keyboard(k, x, y):
    global time_step
    if k==b' ' and time_step < len(history):
        time_step += 1
        glutPostRedisplay()

def special_keys(key, x, y):
    global scroll_x
    scroll_step = 50
    if key == GLUT_KEY_LEFT:
        scroll_x += scroll_step
        glutPostRedisplay()
    elif key == GLUT_KEY_RIGHT:
        scroll_x -= scroll_step
        glutPostRedisplay()


def init():
    glClearColor(1,1,1,1)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluOrtho2D(0, W, 0, H)
    glMatrixMode(GL_MODELVIEW)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowSize(W, H)
    glutCreateWindow(b"RNN Grid Visualizer")
    glutDisplayFunc(display)
    
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)

    init()
    glutMainLoop()

if __name__ == "__main__":
    main()
