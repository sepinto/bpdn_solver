"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")
"""
from __future__ import division
import numpy as np


def QuantizeUniform(aNum, nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    if nBits == 0: return 0
    limit = 1 << nBits
    limit2 = limit >> 1
    # The overload level of the quantizer should be 1.0
    aQ = (int)(((limit - 1) * abs(aNum) + 1) / 2)
    if aQ >= limit2: aQ = limit2 - 1
    if aNum >= 0:
        return aQ
    else:
        return aQ | limit2


        ### Problem 1.a.i ###


def DequantizeUniform(aQuantizedNum, nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """
    if nBits == 0: return 0
    sign_place = aQuantizedNum >> (nBits - 1)
    s = 1 - (sign_place << 1)
    num = aQuantizedNum & ((1 << nBits - 1) - 1)
    return s * 2 * num / ((1 << nBits) - 1)


def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    limit = 1 << nBits;
    limit2 = limit >> 1;

    # The overload level of the quantizer should be 1.0
    # x = |x|
    code = np.absolute(aNumVec)
    # x = ((2^R+1)*x+1)/2
    code = ((limit - 1) * code + 1) / 2
    code = code.astype(np.int32)
    # x = x > 2^R-1 ? 2^R-1 : x
    code[code > limit2 - 1] = limit2 - 1
    code[aNumVec < 0] |= limit2
    return code


def vDequantizeUniform(aQNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """
    # Converts the sign bit to a +/- 1
    sign_place = aQNumVec >> (nBits - 1)
    s = 1 - (sign_place << 1)
    # Extracts the remaining numbers
    num = aQNumVec & ((1 << nBits - 1) - 1)
    return s * 2 * num / ((1 << nBits) - 1)


def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    nBits = ((1 << nScaleBits) - 1 + nMantBits)
    # Uniform Quantization
    limit = 1 << nBits
    limit2 = limit >> 1
    # The overload level of the quantizer should be 1.0
    aQ = (int)(((limit - 1) * abs(aNum) + 1) / 2)
    if aQ >= limit2: aQ = limit2 - 1

    # Find the amount of shift
    nlz = nBits - 1 - aQ.bit_length();
    lz_lim = (1 << nScaleBits) - 1

    # Only room for lz_lim shifting
    if nlz > lz_lim: return lz_lim
    return nlz


def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    nBits = ((1 << nScaleBits) - 1 + nMantBits)
    # Uniform Quantization
    limit = 1 << nBits
    limit2 = limit >> 1
    # The overload level of the quantizer should be 1.0
    aQ = (int)(((limit - 1) * abs(aNum) + 1) / 2)
    if aQ >= limit2: aQ = limit2 - 1

    # Mask the number so that we only get nMantBits-1 bits (excluding the sign bit) 
    mask = (1 << nMantBits - 1) - 1

    if scale == (1 << nScaleBits) - 1:
        # If the number of leading zeros is the limit, just mask. We can't
        # do the leading 1 trick.
        mantissa = (mask & aQ)
    else:
        mantissa = mask & ((aQ << scale) >> (nBits - nMantBits - 1))
    if aNum < 0: mantissa = mantissa | (1 << (nMantBits - 1))
    return mantissa


'''
An efficient means of doing floating point quantization that doesn't involve repeating the same calculations
for the scale and mantissa bits.
'''


def FPQuantization(aNum, nScaleBits=3, nMantBits=5):
    nBits = ((1 << nScaleBits) - 1 + nMantBits)
    # Uniform Quantization
    limit = 1 << nBits
    limit2 = limit >> 1
    # The overload level of the quantizer should be 1.0
    aQ = (int)(((limit - 1) * abs(aNum) + 1) / 2)
    if aQ >= limit2: aQ = limit2 - 1
    # Find the amount of shift
    nlz = nBits - 1 - aQ.bit_length();
    # Only room for lz_lim shifting
    lz_lim = (1 << nScaleBits) - 1
    if nlz > lz_lim: nlz = lz_lim


    # Mask the number so that we only get nMantBits-1 bits (excluding the sign bit) 
    mask = (1 << nMantBits - 1) - 1
    if nlz == lz_lim:
        # If the number of leading zeros is the limit, just mask. We can't
        # do the leading 1 trick.
        mantissa = (mask & aQ)
    else:
        mantissa = mask & ((aQ << nlz) >> (nBits - nMantBits - 1))
    if aNum < 0: mantissa = mantissa | (1 << (nMantBits - 1))

    return (nlz, mantissa)


def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a signed fraction for floating-point scale and mantissa given 
    specified scale and mantissa bits
    """
    nBits = ((1 << nScaleBits) - 1 + nMantBits)
    sign_place = mantissa >> (nMantBits - 1)
    s = 1 - (sign_place << 1)
    # Return the leading 1 (overwriting the sign bit)
    if scale < (1 << nScaleBits) - 1:
        mantissa = (mantissa | (1 << nMantBits - 1))
    else:
        mantissa = (mantissa & ((1 << nMantBits - 1) - 1)) << 1
    mantissa = (mantissa << 1) + 1
    mantissa = mantissa << (nBits - nMantBits - 2) >> scale
    return s * 2 * mantissa / ((1 << nBits) - 1)


def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a signed fraction aNum given 
    nScaleBits scale bits and nMantBits mantissa bits
    """
    nBits = ((1 << nScaleBits) - 1 + nMantBits)
    # Uniform Quantization
    limit = 1 << nBits
    limit2 = limit >> 1
    # The overload level of the quantizer should be 1.0
    aQ = (int)(((limit - 1) * abs(aNum) + 1) / 2)
    if aQ >= limit2: aQ = limit2 - 1

    # Mask the number so that we only get nMantBits-1 bits (excluding the sign bit) 
    mask = (1 << nMantBits - 1) - 1
    # Undo the scale
    mantissa = mask & (aQ >> (nBits - nMantBits - scale))

    if aNum < 0: return mantissa | (1 << (nMantBits - 1))

    return mantissa


def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a signed fraction for block floating-point scale and mantissa 
    given specified scale and mantissa bits
    """
    scaleLim = (1 << nScaleBits) - 1
    nBits = (scaleLim + nMantBits)
    sign_place = mantissa >> (nMantBits - 1)
    s = 1 - (sign_place << 1)
    # Clear the sign bit
    mantissa = (mantissa & ~(1 << nMantBits - 1))

    if mantissa != 0:
        if scale == scaleLim:
            mantissa = mantissa << (nBits - nMantBits - scale)
        else:
            mantissa = (mantissa << 1) + 1
            if nBits - nMantBits - 1 - scale > 0:
                mantissa = mantissa << (nBits - nMantBits - 1 - scale)
    else:
        mantissa = mantissa << (nBits - nMantBits - scale)

    return s * 2 * mantissa / ((1 << nBits) - 1)


def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of signed 
    fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    pos = np.greater_equal(aNumVec, np.zeros(len(aNumVec)))

    nBits = ((1 << nScaleBits) - 1 + nMantBits)
    # Uniform Quantization
    limit = 1 << nBits
    limit2 = limit >> 1

    # The overload level of the quantizer should be 1.0
    # x = |x|
    code = np.absolute(aNumVec)
    # x = ((2^R+1)*x+1)/2
    code = ((limit - 1) * code + 1) / 2
    # x = (int)x
    code = code.astype(np.int32)
    # x = x > 2^R-1 ? 2^R-1 : x
    code[code > limit2 - 1] = limit2 - 1
    # Add the sign bit if negative


    # Mask the number so that we only get nMantBits-1 bits (excluding the sign bit) 
    code = np.bitwise_and(np.right_shift(code, nBits - nMantBits - scale), (1 << nMantBits - 1) - 1)
    # Add the sign bit if negative
    code[aNumVec <= 0] |= 1 << (nMantBits - 1)
    return code


def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of signed fractions for block floating-point scale and vector of 
    block floating-point mantissas given specified scale and mantissa bits
    """
    scaleLim = (1 << nScaleBits) - 1
    nBits = (scaleLim + nMantBits)
    sign_place = np.right_shift(mantissaVec, (nMantBits - 1))
    s = 1 - (sign_place << 1)
    # Clear the sign bit
    mantissaVec = np.bitwise_and(mantissaVec, ~(1 << nMantBits - 1))
    if scale == scaleLim:
        mantissaVec = mantissaVec << (nBits - nMantBits - scale)
    else:
        mantissaVec[mantissaVec != 0] = (mantissaVec[mantissaVec != 0] << 1) + 1
        if nBits - nMantBits - 1 - scale > 0:
            mantissaVec[mantissaVec != 0] = (mantissaVec[mantissaVec != 0] << (nBits - nMantBits - 1 - scale))
    mantissaVec[mantissaVec == 0] = np.left_shift(mantissaVec[mantissaVec == 0], (nBits - nMantBits - scale))

    return s * 2 * mantissaVec / ((1 << nBits) - 1)

