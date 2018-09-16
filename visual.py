import numpy as np
import scipy as sp
import scipy.interpolate
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
import visvis
from matplotlib.mlab import griddata
from scipy import interpolate
# 2D grid construction

def plot_map(data, point = False, path = False, start = 0, stop = 0, pth = 0):
    """data is a list of tuples <lat, long, ele>"""
# transform to numpy arrays
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])
    z = np.array([i[2] for i in data])
    spline = sp.interpolate.Rbf(x, y, z, function='thin-plate')
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(xi, yi)
    # interpolation
    Z = spline(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.hold(True)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True);
    #plt.show()

    if point == True:
        pass

    if path == True:
        for i in range(len(pth)):
            pth[i] = data[i]
        xp = np.array([i[0] for i in pth])
        yp = np.array([i[1] for i in pth])
        zp = np.array([i[2] for i in pth])
        ax.scatter(xp, yp, zp, color='black');

        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.0001)
        pass

    else:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.0001)


def alter_map(data, pth):
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])
    z = np.array([i[2] for i in data])
    spline = sp.interpolate.Rbf(x, y, z, function='thin-plate')
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(xi, yi)
    # interpolation
    Z = spline(X, Y)

    # contours = plt.contour(X, Y, Z, 3, colors='black')
    # plt.clabel(contours, inline=True, fontsize=8)
    #
    # plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
    #            cmap='RdGy', alpha=0.5)
    # plt.colorbar();
    plt.hold(True)
    contours = plt.contour(X, Y, Z, cmap=cm.jet, color='k', linewidth=1);
    plt.clabel(contours, inline=True, fontsize=8)
    #plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',cmap=cm.jet, alpha=0.5)
    #plt.colorbar();

    for i in range(len(pth)):
        pth[i] = data[i]
    xp = np.array([i[0] for i in pth])
    yp = np.array([i[1] for i in pth])
    zp = np.array([i[2] for i in pth])
    plt.scatter(xp, yp, zp, color='black');

    plt.show()