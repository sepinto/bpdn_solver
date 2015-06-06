"""
window.py -- Defines functions to window an array of data samples
Created by Chet Gnegy
"""

from __future__ import division
import numpy as np

######################################## SINE WINDOWS ####################################################
global sine_win_dict
sine_win_dict = {}

# Generates a Sine Window of length N
def GenerateSineWindow(N):
    global sine_win_dict
    if N not in sine_win_dict.keys():
        c = np.pi / N
        offset = 0.5 * np.pi / N
        sine_win_dict[N] = np.sin(c * np.arange(N) + offset)
    return sine_win_dict[N]


def SineWindow(data):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """
    return np.multiply(data, GenerateSineWindow(len(data)))


def SinePower(N):
    return (1.0 / N) * np.sum(np.power(SineWindow(np.ones(N)), 2.0))


######################################## HANNING WINDOWS ####################################################
global hann_win_dict
hann_win_dict = {}

# Generates a Hann Window of length N
def GenerateHannWindow(N):
    global hann_win_dict
    if N not in hann_win_dict.keys():
        c = 2.0 * np.pi / N
        offset = np.pi / N
        hann_win_dict[N] = 0.5 - 0.5 * np.cos(c * np.arange(N) + offset)
    return hann_win_dict[N]


def HanningWindow(data):
    """
    Returns a copy of the dataSampleArray Sine-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """
    return np.multiply(data, GenerateHannWindow(len(data)))


def HanningPower(N):
    return (1.0 / N) * np.sum(np.power(HanningWindow(np.ones(N)), 2.0))

######################################## RECTANGULAR WINDOWS ####################################################
def RectangleWindow(data):
    return data

def RectanglePower(N):
    return (1.0 / N) * np.sum(np.power(RectangleWindow(np.ones(N)), 2.0))

# The window function which is used throughout project. Just swap in different windows here.
def Window(data):
    return RectangleWindow(data)

def WindowPower(N):
    return RectanglePower(N)