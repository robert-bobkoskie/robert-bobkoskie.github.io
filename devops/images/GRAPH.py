
import sys, time, math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import itertools
from scipy import linalg
    

def ST_Parabola():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = x**2 + 2*x + 2
        a.append(x)
        b.append(y)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = x^2 + 2x + 2$')

    plt.title('PARABOLA (Positive)')
    #plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    #plt.savefig(r"EM_COV_"+str(filename)+"_.png")
    plt.savefig(r"ST_Parabola.png")
    plt.close()

def ST_Neg_Parabola():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = -x**2 + 2*x + 2
        a.append(x)
        b.append(y)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = -x^2 + 2x + 2$')

    plt.title('PARABOLA (Negative)')
    #plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    #plt.savefig(r"EM_COV_"+str(filename)+"_.png")
    plt.savefig(r"ST_Neg_Parabola.png")
    plt.close()

def HC_Parabola():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = x**2 + 2*x + 2
        a.append(x)
        b.append(y)

    fig = plt.figure()
    plt.style.use('dark_background')
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = x^2 + 2x + 2$')
    axes.set_axis_bgcolor('black')

    plt.title('PARABOLA (Positive)')
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"HC_Parabola.png")
    plt.close()

def HC_Neg_Parabola():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = -x**2 + 2*x + 2
        a.append(x)
        b.append(y)

    fig = plt.figure()
    plt.style.use('dark_background')
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = -x^2 + 2x + 2$')
    axes.set_axis_bgcolor('black')

    plt.title('PARABOLA (Negative)')
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"HC_Neg_Parabola.png")
    plt.close()

def ST_Circle():
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi, 100)

    # the radius of the circle
    r = np.sqrt(4.)

    # compute x1 and x2
    a = r*np.cos(theta)
    b = r*np.sin(theta)

    # create the figure
    fig, axes = plt.subplots(1)
    axes.plot(a,b, label='$x^2 + y^2 = 4$')
    axes.set_aspect(1)

    plt.title('CIRCLE')
    plt.legend(loc='center', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"ST_Circle.png")
    plt.close()

def HC_Circle():
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi, 100)

    # the radius of the circle
    r = np.sqrt(4.)

    # compute x1 and x2
    a = r*np.cos(theta)
    b = r*np.sin(theta)

    # create the figure
    fig, axes = plt.subplots(1)
    plt.style.use('dark_background')
    axes.plot(a,b, label='$x^2 + y^2 = 4$')
    axes.set_aspect(1)
    axes.set_axis_bgcolor('black')

    plt.title('CIRCLE')
    plt.legend(loc='center', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"HC_Circle.png")
    plt.close()


def ST_Hyperbola():
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(figsize=plt.figaspect(1))
    axes = Axes3D(fig)

    r=1;
    u=np.linspace(-2,2,200);
    v=np.linspace(0,2*np.pi,60);
    [u,v]=np.meshgrid(u,v);

    a = 1
    b = 1
    c = 1

    x = a*np.cosh(u)*np.cos(v)
    y = b*np.cosh(u)*np.sin(v)
    z = c*np.sinh(u)

    #axes.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')
    axes.plot_surface(x, y, z,  rstride=4, cstride=4, color='w')
    #axes.text2D(0.05, 0.95, '$x^2/a^2 + y^2/b^2 = c$', transform=axes.transAxes)
    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')

    #plt.show()
    plt.title('3-D HYPERBOLA: $x^2/a^2 + y^2/b^2 = c$')
    plt.savefig(r"ST_3D_Hyperbola.png")
    plt.close()

def HC_Hyperbola():
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(figsize=plt.figaspect(1))
    plt.style.use('dark_background')    
    axes = Axes3D(fig)
    axes.set_axis_bgcolor('black')

    r=1;
    u=np.linspace(-2,2,200);
    v=np.linspace(0,2*np.pi,60);
    [u,v]=np.meshgrid(u,v);

    a = 1
    b = 1
    c = 1

    x = a*np.cosh(u)*np.cos(v)
    y = b*np.cosh(u)*np.sin(v)
    z = c*np.sinh(u)

    axes.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')
    #axes.plot_surface(x, y, z,  rstride=4, cstride=4, color='w')
    #axes.text2D(0.05, 0.95, '$x^2/a^2 + y^2/b^2 = c$', transform=axes.transAxes)
    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')

    #plt.show()
    plt.title('3-D HYPERBOLA: $x^2/a^2 + y^2/b^2 = c$')
    plt.savefig(r"HC_3D_Hyperbola.png")
    plt.close()

def ST_Line_Horizontal():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = 3
        a.append(x)
        b.append(y)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = 3$')

    plt.title('LINE (Horizontal)')
    #plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    #plt.savefig(r"EM_COV_"+str(filename)+"_.png")
    plt.savefig(r"ST_Hrz_Line.png")
    plt.close()

def ST_Line_Positive():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = x
        a.append(x)
        b.append(y)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = x$')

    plt.title('LINE (Positive Slope)')
    #plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    #plt.savefig(r"EM_COV_"+str(filename)+"_.png")
    plt.savefig(r"ST_Pos_Line.png")
    plt.close()

def ST_Line_Negative():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = -x
        a.append(x)
        b.append(y)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = -x$')

    plt.title('LINE (Negative Slope)')
    #plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    #plt.savefig(r"EM_COV_"+str(filename)+"_.png")
    plt.savefig(r"ST_Neg_Line.png")
    plt.close()

def HC_Line_Horizontal():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = 3
        a.append(x)
        b.append(y)

    fig = plt.figure()
    plt.style.use('dark_background')
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = 3$')
    axes.set_axis_bgcolor('black')

    plt.title('LINE (Horizontal)')
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"HC_Hrz_Line.png")
    plt.close()

def HC_Line_Positive():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = x
        a.append(x)
        b.append(y)

    fig = plt.figure()
    plt.style.use('dark_background')
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = x$')
    axes.set_axis_bgcolor('black')

    plt.title('LINE (Positive Slope)')
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"HC_Pos_Line.png")
    plt.close()

def HC_Line_Negative():
    a=[]
    b=[]

    for x in range(-50,50,1):
        y = -x
        a.append(x)
        b.append(y)

    fig = plt.figure()
    plt.style.use('dark_background')
    axes = fig.add_subplot(111)
    axes.plot(a,b, label='$y = -x$')
    axes.set_axis_bgcolor('black')

    plt.title('LINE (Negative Slope)')
    plt.legend(loc='upper right', prop={'size':16})

    #plt.show()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(r"HC_Neg_Line.png")
    plt.close()


################################
def NUMPY():
    #############################
    # mdptoolbox
    #############################
    import mdptoolbox
    import mdptoolbox.example

    x = np.arange(27, dtype=np.float).reshape((3, 3, 3))
    np.array_split(x, 3)
    #x = np.indices((3,3))

    print x.shape, x.dtype, x[1,0]
    print x
    #x[:,:,1] = 0    #Middle Col
    #x[:,:1] = 1     #Top Row

    x = np.zeros((2,10,10))
    x = np.zeros((2,4,5))
    x = np.full((2,4,5), 0.01)
    print 'DTYPE', x.dtype

    term_state = []
    for i in range(int(x.shape[0])):
        x[i,0,0]                         = -10.0
        term_state.append( (i,0,0) )

        x[i, 0, x.shape[2]-1]            =  10.0
        term_state.append( (i, 0, int(x.shape[2]-1)) )

        x[i, x.shape[1]-1, 0]            =  10.0
        term_state.append( (i, int(x.shape[1]-1), 0) )

        x[i, x.shape[1]-1, x.shape[2]-1] = -10.0
        term_state.append( (i, int(x.shape[1]-1), int(x.shape[2]-1)) )

        x[i, x.shape[1]/2-1 : x.shape[1]/2+1,
             x.shape[2]/2-1 : x.shape[2]/2+1] = np.NaN
    print '\n================'
    print x
    print term_state
    print x.shape
    for idx, val in np.ndenumerate(x):
        pass
        #start_state = idx
        #print 'HERE', start_state, start_state[1], '--', x.shape[0], x.shape[1], x.shape[2]
        #print ((x.shape[2]*start_state[1]) + start_state[2] + start_state[0]) +\
        #      ((x.shape[1]*x.shape[2]*start_state[0]) - start_state[0])

    print '\n================'
    a = np.random.randint(0,5,(3,3))
    ua, uind = np.unique(a,return_inverse=True)
    count = np.bincount(uind)
    print a
    print '\n', ua, count


#############################
def main():
    #NUMPY()
    ST_Parabola()
    ST_Neg_Parabola()
    ST_Circle()
    ST_Hyperbola()
    ST_Line_Horizontal()
    ST_Line_Positive()
    ST_Line_Negative()

    HC_Parabola()
    HC_Neg_Parabola()
    HC_Circle()
    HC_Hyperbola()
    HC_Line_Horizontal()
    HC_Line_Positive()
    HC_Line_Negative()

if __name__ == '__main__':
    main()
#############################
