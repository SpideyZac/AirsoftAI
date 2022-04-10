import numpy as np
cimport numpy as np

cimport cython

cpdef bint collision(double x1, double y1, double x2, double y2, double width1, double height1, double width2, double height2):
    x_collision = (np.fabs(x1 - x2) * 2) < (width1 + width2)
    y_collision = (np.fabs(y1 - y2) * 2) < (height1 + height2)
    return (x_collision and y_collision)

ctypedef np.float64_t DTYPE_t
cpdef np.ndarray[DTYPE_t, ndim=1] forward(double x, double y, double rot):
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros(2)
    out[0] = np.sin(rot)
    out[1] = np.cos(rot)
    return out

cpdef tuple ray(double x, double y, double rot, list walls, int oppx, int oppy):
    cdef int num_walls = len(walls)
    cdef int i
    cdef bint hit_player = False
    cdef bint collided = False
    cdef int distancee = 0
    cdef np.ndarray[DTYPE_t, ndim=1] f = forward(x, y, rot)
    cdef int maxe = 20

    while not collided and distancee <= maxe:
        for i in range(num_walls):
            collided = collision(x + f[0] * distancee, y + f[1] * distancee, walls[i][0], walls[i][1], 0.5, 0.5, 1, 1)
            if collided: break
        if not hit_player:
            hit_player = collision(x + f[0] * distancee, y + f[1] * distancee, oppx, oppy, 1, 1, 1, 1)
        if not collided:
            distancee += 1

    return (distancee, hit_player, x + f[0] * distancee, y + f[1] * distancee)

cpdef double distance(double x1, double y1, double x2, double y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)