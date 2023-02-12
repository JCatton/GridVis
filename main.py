import sys
import numpy as np
import pyqtgraph.opengl as gl
from opensimplex import OpenSimplex
from PyQt6.QtWidgets import QApplication, QWidget
from pyqtgraph.Qt import QtCore, QtGui
from time import perf_counter


class Terrain:
    def __init__(self):
        self.end_time = None
        self.app = QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.setGeometry(0, 110, 1366, 768)
        self.w.show()
        self.w.setWindowTitle('Terrain')
        self.w.setCameraPosition(distance=50, elevation=5)

        grid = gl.GLGridItem()
        grid.scale(2, 2, 2)
        self.w.addItem(grid)

        self.nsteps = 0.1
        self.grid_points = 100
        self.ypoints = np.linspace(-20, 22, self.grid_points)
        self.xpoints = np.linspace(-20, 22, self.grid_points)
        self.nfaces = len(self.ypoints)
        self.timeoffset = 0
        self.xdamp = 2
        self.ydamp = 0.2

        self.tmp = OpenSimplex(np.random.randint(0, 1_000_000))

        verts = np.array([
                [
                        x, y, self.tmp.noise2(x=n / self.xdamp + self.timeoffset, y=m / self.ydamp)
                ] for n, x in enumerate(self.xpoints) for m, y in enumerate(self.ypoints)
        ], dtype=np.float32)
        """
        Treating faces as triangles within square grid.
        eg.
        2 - 0
        |   |
        3 - 1
        Two faces, 0,1,2 and 1,2,3. Since have added to our grid all the x values per y value,
        to add 1 to the y value, we add self.nfaces. Hence why yoff is m * self.nfaces.
        Our faces 1st value is just everypoint in the grid apart from the last x, and the 2nd
        value is that with an increment of y, and the 3rd value is the with an increment of 1
        in the x direction.
        """

        faces = []
        colors = []
        for m in range(self.nfaces - 1):
            yoff = m * self.nfaces
            for n in range(self.nfaces - 1):
                x_val = n + yoff
                faces.append([x_val, x_val + self.nfaces, x_val + self.nfaces + 1])
                faces.append([x_val, x_val + 1, x_val + self.nfaces + 1])
                colorratio = n / self.nfaces
                colors.append([colorratio, 1 - colorratio, m / self.nfaces, 0.9])
                colors.append([colorratio, 1 - colorratio, m / self.nfaces, 0.5])
        faces = np.asarray(faces)
        colors = np.asarray(colors)

        self.mesh1 = gl.GLMeshItem(
                vertexes=verts,
                faces=faces, faceColors=colors,
                smooth=False, drawEdges=True,
                computeNomrals=False
        )
        self.mesh1.setGLOptions('additive')
        self.w.addItem(self.mesh1)
        return None

    def update(self):

        # Vectorized Code
        # =====================================
        X, Y = np.meshgrid(self.xpoints, self.ypoints, indexing='ij')
        Xs, Ys = X.flatten(), Y.flatten()
        verts = np.empty(shape=(len(self.xpoints) * len(self.ypoints), 3))
        verts[:, 0], verts[:, 1] = Xs, Ys
        # Creating custom z values from the grid
        verts[:, 2] = 1 * np.sin(Xs + self.timeoffset) + 1 * np.cos(Ys + 2 * self.timeoffset)
        verts[:, 2] += 0.5 * np.sin(Xs - 2 * self.timeoffset)
        # verts[:, 2] = gaussian2d(((Xs - self.timeoffset) % len(self.xpoints)), Ys, 20, 0, 0, 1)
        # verts[:, 2] += gaussian2d(((Xs - self.timeoffset) % len(self.xpoints)), (Ys - self.timeoffset) % len(self.ypoints), 40, 0, 0, 5)
        # verts[:, 2] += gaussian2d(((Xs - self.timeoffset) % len(self.xpoints)),
        #                           (Ys + self.timeoffset) % len(self.ypoints), 40, 0, 0, 10)

        faces_1 = np.empty(shape=((self.nfaces - 1) * (self.nfaces - 1), 3), dtype=int)
        faces_2 = np.copy(faces_1)
        colors = np.empty(shape=((self.nfaces - 1) * (self.nfaces - 1), 4), dtype=np.float32)

        mvals = np.arange(self.nfaces - 1)
        yoffs = mvals * self.nfaces
        xvals = np.asarray([mvals + offset for offset in yoffs]).flatten()
        faces_1[:, 0], faces_1[:, 1], faces_1[:, 2] = xvals, xvals + self.nfaces, xvals + self.nfaces + 1
        faces_2[:, 0], faces_2[:, 1], faces_2[:, 2] = xvals, xvals + 1, xvals + self.nfaces + 1
        faces = np.append(faces_1, faces_2, axis=0)

        n = xvals % self.nfaces
        temp, m = np.meshgrid(mvals, mvals)
        colors[:, 0] = n / self.nfaces
        colors[:, 1] = 1 - n / self.nfaces
        colors[:, 2] = m.flatten() / self.nfaces
        colors[:, 3] = 0.8
        colors = np.append(colors, colors, axis=0)
        # =====================================

        # Testing
        # faces_1 = np.empty(shape=((nfaces - 1) * (nfaces - 1), 3), dtype=int)
        # faces_2 = np.copy(faces_1)
        # # colors = np.empty(shape=(self.nfaces - 1, self.nfaces - 1, 3), dtype=np.float32)
        # mvals = np.arange(nfaces - 1)
        # yoffs = mvals * nfaces
        # xvals = np.asarray([mvals + offset for offset in yoffs]).flatten()
        # faces_1[:,0], faces_1[:,1], faces_1[:,2] = xvals, xvals + nfaces, xvals + nfaces + 1
        # faces_2[:,0], faces_2[:,1], faces_2[:,2] = xvals, xvals + 1, xvals + nfaces + 1
        # faces = np.append(faces_1, faces_2, axis=0)

        # Old Code
        # =====================================
        # verts = np.array([
        #         [
        #                 x, y, 1 * np.sin(x + self.timeoffset) + 0.5 * np.sin(x - 2 * self.timeoffset) + 1 * np.cos(y + 2 * self.timeoffset)
        #         ] for n, x in enumerate(self.xpoints) for m, y in enumerate(self.ypoints)
        # ], dtype=np.float32)
        # faces = []
        # colors = []
        # for m in range(self.nfaces - 1):
        #     yoff = m * self.nfaces
        #     for n in range(self.nfaces - 1):
        #         x_val = n + yoff
        #         faces.append([x_val, x_val + self.nfaces, x_val + self.nfaces + 1])
        #         faces.append([x_val, x_val + 1, x_val + self.nfaces + 1])
        #         colorratio = n / self.nfaces
        #         colors.append([colorratio, 1 - colorratio, m / self.nfaces, 0.9])
        #         colors.append([colorratio, 1 - colorratio, m / self.nfaces, 0.5])
        # faces = np.asarray(faces)
        # colors = np.asarray(colors)
        # colors = []
        # for m in range(self.nfaces - 1):
        #     for n in range(self.nfaces - 1):
        #         colorratio = n / self.nfaces
        #         colors.append([colorratio, 1 - colorratio, m / self.nfaces, 0.9])
        #         colors.append([colorratio, 1 - colorratio, m / self.nfaces, 0.5])
        # colors = np.asarray(colors)
        # =====================================

        # Adding the meshdata
        self.mesh1.setMeshData(
                vertexes=verts,
                faces=faces, faceColors=colors
        )
        self.timeoffset -= 0.01
        if self.timeoffset < -5:
            if self.end_time is None:
                self.end_time = perf_counter()
            print(f'Run took {self.end_time-self.start_time} Seconds')

    def start(self):
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        self.app.exec()



    def animation(self, interval: int = 100):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(interval)
        self.start_time = perf_counter()
        self.start()
        self.update()

@np.vectorize
def gaussian2d(x, y, A, x_0, y_0, sig):
    return A * np.exp((-(x - x_0) * (x - x_0) - (y - y_0) * (y - y_0)) / (2 * sig)) / (2 * np.pi * sig * sig)


if __name__ == '__main__':
    t = Terrain()
    t.animation(1)

# Timing, Vectorized 7.15 seconds
# Non-vect noise, 265.5 Seconds..
# Non-vect sin waves # 86.3 seconds

# Timing w/o normal computation Vectorized 2.9 seconds
# w/o normal computation Non-vect sin waves  53.9 seconds