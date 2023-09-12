#!/usr/bin/env python3
"""My code for Part A"""

from matplotlib import RcParams
import numpy as np
import matplotlib.pyplot as plt



def myLS1(xvals, yvals):
    """
    linear least squares
    """
    x = np.array([xvals])
    y = np.array([yvals]).T
    o = np.array([np.ones(len(xvals))])
    A = np.concatenate((x, o), axis=0).T
    # print(A)
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def myLS2(xvals, yvals):
    """
    Least squares 2nd order
    """
    x = np.array([xvals])
    x2 = x*x
    y = np.array([yvals]).T
    o = np.array([np.ones(len(xvals))])
    A = np.concatenate((x2, x, o), axis=0).T
    # print(A)
    a, b , c = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b, c

def myLS3(xvals, yvals):
    """
    Least squares 3rd order
    """
    x = np.array([xvals])
    x2 = x*x
    x3 = x*x*x
    y = np.array([yvals]).T
    o = np.array([np.ones(len(xvals))])
    A = np.concatenate((x3, x2, x, o), axis=0).T
    # print(A)
    a, b , c ,d= np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b, c,d



def myLS4(xvals, yvals):
    """
    Least squares 4th order
    """
    x = np.array([xvals])
    x2 = x*x
    x3 = x*x*x
    x4 = x*x*x*x
    y = np.array([yvals]).T
    o = np.array([np.ones(len(xvals))])
    A = np.concatenate((x4, x3, x2, x, o), axis=0).T
    # print(A)
    a, b , c ,d , e= np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b, c,d, e



def my1LS(xvals, yvals):
    """
    Least squares 1/x
    """
    x = np.array([xvals])
    y = np.array([yvals]).T
    o = np.array([np.ones(len(xvals))])
    A = np.concatenate((1/x, o), axis=0).T
    print(A)
    b , c = np.linalg.lstsq(A, y, rcond=None)[0]
    return b, c


def myLnLS(xvals, yvals):
    """
    Least squares -ln (x+a) + b
    """
    a_arr = np.arange(0, 5, 0.1)
    b_arr = np.arange(-10, 10, 0.1)
    x = np.array([xvals])
    # y = np.array([yvals]).T

    # i_arr = np.arange(0, len(x))
    # np.random.shuffle(i_arr)

    leastS = 1000000
    a = None
    b = None
    for aa in a_arr:
        print(f"cur aa is {aa}")
        for bb in b_arr:
            curSquare = 0
            for i in range (0, len(xvals)):
                cury = -np.log(xvals[i] + aa) + bb
                cur = (yvals[i] - cury)
                cur = cur*cur
                curSquare += cur
                if curSquare > leastS:
                    break
            
            if curSquare < leastS:
                # print(f"cur least squares = {curSquare}")
                leastS = curSquare
                a = aa
                b = bb
                print(f"cur least squares = {curSquare}, a {a}, b {b}")
                if leastS < 0.01:
                    return a, b
    return a, b

def myLnLSv2(xvals, yvals):
    """
    Least squares -ln (x+a) + b
    """
    a_arr = np.arange(0, 3, 0.1)
    b_arr = np.arange(-10, 10, 0.1)
    c_arr = np.arange(-10, 0, 0.1)
    x = np.array([xvals])
    # y = np.array([yvals]).T

    # i_arr = np.arange(0, len(x))
    # np.random.shuffle(i_arr)

    leastS = 1000
    a = None
    b = None
    for aa in a_arr:
        print(f"cur aa is {aa}")
        for bb in b_arr:
            for cc in c_arr:
                curSquare = 0
                for i in range (0, len(xvals)):
                    cury = cc * np.log(xvals[i] + aa) + bb
                    cur = (yvals[i] - cury)
                    cur = cur*cur
                    curSquare += cur
                    if curSquare > leastS:
                        break
                
                if curSquare < leastS:
                    # print(f"cur least squares = {curSquare}")
                    leastS = curSquare
                    a = aa
                    b = bb
                    c = cc
                    print(f"cur least squares = {curSquare}, a {a}, b {b}")
                    if leastS < 0.01:
                        return a, b, c
    return a, b, c



def myLogLS(xvals, yvals):
    """
    Least squares -log (x+a) + b
    """
    a_arr = np.arange(0, 10, 0.05)
    b_arr = np.arange(-10, 10, 0.05)
    x = np.array([xvals])

    leastS = 1000000
    a = None
    b = None
    for aa in a_arr:
        for bb in b_arr:
            curSquare = 0
            for i in range (0, len(xvals)):
                cury = -np.log10(xvals[i] + aa) + bb
                cur = (yvals[i] - cury)
                cur = cur*cur
                curSquare += cur
                if curSquare > leastS:
                    break

            if curSquare < leastS:
                # print(f"cur least squares = {curSquare}")
                leastS = curSquare
                a = aa
                b = bb
                print(f"cur least squares = {curSquare}, a {a}, b {b}")

    return a, b



def my1LSv2(xvals, yvals):
    """
    Least squares 1/x + b
    """
    a_arr = np.arange(-10, 10, 1)
    b_arr = np.arange(-10, 10, 1)
    o_arr = np.arange(-10, 10, 1)
    x = np.array([xvals])
    # y = np.array([yvals]).T

    # i_arr = np.arange(0, len(x))
    # np.random.shuffle(i_arr)

    ls = (-3, -3)
    leastS = 1000
    a = None
    b = None
    for aa in a_arr:
        print(f"cur a = {aa}")
        for bb in b_arr:
            for oo in o_arr:
                curSquare = 0
                for i in range (0, len(xvals)):
                    cury = aa/(xvals[i]+oo) + bb
                    cur = (yvals[i] - cury) **2
                    curSquare += cur
                    if curSquare > leastS:
                        break
                if curSquare < leastS:
                    
                    leastS = curSquare
                    a = aa
                    b = bb
                    o = oo
                    print(f"cur least squares = {curSquare}, a {a}, b {b}, o {o}")

    return a, b, o


def my1LS1(xvals, yvals):
    """
    Least squares a0/(x+a1)  + b + c*(x)
    """
    a0_arr = np.arange(42, 46, 0.1)
    a1_arr = np.arange(7, 9, 0.1)
    b_arr = np.arange(-2, 1, 0.1)
    
    c0_arr = np.arange(0, 1, 0.01)
    x = np.array([xvals])
    # y = np.array([yvals]).T

    # i_arr = np.arange(0, len(x))
    # np.random.shuffle(i_arr)

    ls = (-3, -3)
    leastS = 10000000
    a0 = None
    a1 = None
    b = None
    c = None


    for aa0 in a0_arr:
        print(aa0)
        for aa1 in a1_arr:
            for bb in b_arr:
                for cc0 in c0_arr:
                    curSquare = 0
                    for i in range (0, len(xvals)):
                        cury = aa0/(xvals[i]+aa1)  + bb + cc0*(xvals[i])
                        cur = (yvals[i] - cury) **2
                        curSquare += cur
                        if curSquare > leastS:
                            break
                    if curSquare < leastS:
                        # print("curSquare", curSquare)
                        leastS = curSquare
                        a0 = aa0
                        a1 = aa1
                        b = bb
                        c = cc0

                            # print(f"cur least squares = {curSquare}, a {a}, b {b}, o {o}")

    return a0, a1, b, c




def myIRLS1(xvals, yvals, Titer=4):
    
    # print(np.shape(yvals)) #shape should be (bignum, 1) for col vector
    for iter in range (0, Titer):
        # print(iter)
        # print(len(xvals))
        # print("FISH")
        m, c = myLS1(xvals, yvals)
        
        error = [None] * len(xvals)
        for i in range (len(xvals)-1, -1, -1):
            error[i] = m * xvals[i] + c - yvals[i] 

        maxerror = max(error)
        for i in range (len(xvals)-1, 0, -1):
            if (abs(error[i])> 0.5 * maxerror):
                yvals = np.delete(yvals, i)
                xvals = np.delete(xvals, i)
                error = np.delete(error, i)

        # plt.figure(figsize=(5, 5))
        # plt.plot(xvals, yvals, '.', alpha=0.2)
        # plt.plot(xvals, error, '-', alpha=0.2)

        # plt.show()

    return m, c, error, xvals


def myIRLS4(xvals, yvals, Titer=5):
    
    # print(np.shape(yvals)) #shape should be (bignum, 1) for col vector
    for iter in range (0, Titer):
        # print(iter)
        # print(len(xvals))
        a, b, c, d, e = myLS4(xvals, yvals)
        
        error = [None] * len(xvals)
        for i in range (len(xvals)-1, -1, -1):
            curx = xvals[i]
            error[i] = a * curx**4 + b *curx**3  + c*curx**2 + d*curx + e - yvals[i] 

        maxerror = max(error)
        for i in range (len(xvals)-1, 0, -1):
            if (abs(error[i])> 0.5 * maxerror):
                yvals = np.delete(yvals, i)
                xvals = np.delete(xvals, i)
                error = np.delete(error, i)

        # plt.figure(figsize=(5, 5))
        # plt.plot(xvals, yvals, '.', alpha=0.2)
        # plt.plot(xvals, error, '-', alpha=0.2)

        # plt.show()

    return a, b, c, d, e


def myRootLS(xvals, yvals):
    """
    Least squares    a sqrt (x +o)  + b
    """
    a_arr = np.arange(-10, 10, 1)
    b_arr = np.arange(-10, 10, 1)
    o_arr = np.arange(-10, 10, 1)
    x = np.array([xvals])
    # y = np.array([yvals]).T

    # i_arr = np.arange(0, len(x))
    # np.random.shuffle(i_arr)

    leastS = 1000
    a = None
    b = None
    for aa in a_arr:
        print(f"cur a = {aa}")
        for bb in b_arr:
            for oo in o_arr:
                curSquare = 0
                for i in range (0, len(xvals)):
                    cury = aa * np.sqrt(xvals[i] + oo) + bb
                    cur = (yvals[i] - cury) **2
                    curSquare += cur
                    if curSquare > leastS:
                        break
                if curSquare < leastS:
                    
                    leastS = curSquare
                    a = aa
                    b = bb
                    o = oo
                    print(f"cur least squares = {curSquare}, a {a}, b {b}, o {o}")

    return a, b, o


