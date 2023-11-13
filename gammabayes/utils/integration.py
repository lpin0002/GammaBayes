import numpy as np
from scipy import special

def _basic_simps(y, x):
    """
    Perform vectorized Simpson integration. 
        
    Not suitable for potentially numerically unstable x values.

    Parameters:
    - y: Array of function values.
    - x: Array of x values (evenly spaced in log10 space).

    Returns:
    - result: Simpson's rule integration result.
    """
    if len(y) != len(x):
        raise ValueError("Input arrays y and x must have the same length.")

    # Calculate the step size
    h = np.diff(x)
    length_h = len(h)

        
    
    if length_h%2==0:
        # even case
        result  = (h[:-1:2]+h[1::2])*(
            (2 - h[1::2]/h[:-1:2]) * y[:-1:2] + \
            (h[:-1:2] + h[1::2])**2 / (h[:-1:2]*h[1::2]) * y[1:-1:2] + \
            (2 - h[:-1:2]/h[1::2]) * y[2::2]
            )
        result = np.sum(result)

    
    else:
        # odd case
        result  = (h[:-1:2]+h[1::2])*(
            (2 - h[1::2]/h[:-1:2]) * y[:-2:2] + \
            (h[:-1:2] + h[1::2])**2 / (h[:-1:2]*h[1::2]) * y[1:-1:2] + \
            (2 - h[:-1:2]/h[1::2]) * y[2::2]
            )

        result = np.sum(result)
        _alpha  = (2*h[-1]**2 + 3*h[-1]*h[-2])/(h[-2] + h[-1])
        _beta   = (h[-1]**2 + 3*h[-1]*h[-2])/h[-2]
        _eta    = h[-1]**3 / (h[-2] * (h[-2] + h[-1]))

        result+= _alpha*y[-1]+_beta*y[-2]-_eta*y[-3]


    return result/6

def logspace_trapz(logy, x, axis=-1):
    h = np.diff(x)

    logfkm1    = logy[:-1]
    logfk      = logy[1:]

    trapz_int = special.logsumexp(np.logaddexp(logfkm1, logfk)+np.log(h) - np.log(2))

    return trapz_int


def logspace_simpson(logy, x, axis=-1):
    """
    Perform vectorized Simpson integration for log integrand values. 
        
    Not suitable for potentially numerically unstable x values.

    Parameters:
    - logy: Array of function values.
    - x: Array of x values (evenly spaced in log10 space).

    Returns:
    - result: Simpson's rule integration result.
    """
    if logy.shape[axis] != len(x):
        raise ValueError("Input arrays y and x must have the same length.")

    # Calculate the step size
    
    if len(x)>2:
        h = np.diff(x)
        length_h = len(h)
        # Sorry for naming convention I couldn't think of anything particularly better.
            # They refer to the indices that each array corresponds to when calculating
            # an integral when using the simpson algorithm
        logy_even_small = logy.take( indices=range(0,logy.shape[axis]-2,2), axis=axis )
        logy_odd        = logy.take( indices=range(1,logy.shape[axis]-1,2), axis=axis )
        logy_even_large = logy.take( indices=range(2,logy.shape[axis]-0,2), axis=axis )

        heven   = np.expand_dims(h[:-1:2],  axis=(*np.delete(np.arange(len(logy_even_small.shape)), axis),)) 
        hodd    = np.expand_dims(h[1::2],   axis=(*np.delete(np.arange(len(logy_even_small.shape)), axis),))

        term1 = np.log(2 - hodd/heven) + logy_even_small
        term2 = np.log((heven + hodd)**2 / (heven*hodd)) + logy_odd
        term3 = np.log(2 - heven/hodd) + logy_even_large

        result  = special.logsumexp(np.log(heven+hodd)+special.logsumexp([
                term1,
                term2, 
                term3
            ], axis=0), axis=axis)

        if length_h%2!=0:
            # Odd case, final interval 
            _alpha  = (2*h[-1]**2 + 3*h[-1]*h[-2])/(h[-2] + h[-1])
            _beta   = (h[-1]**2 + 3*h[-1]*h[-2])/h[-2]
            _eta    = h[-1]**3 / (h[-2] * (h[-2] + h[-1]))


            # To account for numerically unstable values we make them less unstable
                # by multiplying all the values such that the last entry in the 
                # relevant axis becomes 1
                # It isn't any guarantee though so be wary. Best solution if you're
                # worried it to simply have an even number of intervals 
                # i.e. length_h%2==0
            logy_m3     = logy.take( indices=-3, axis=axis )
            logy_m2     = logy.take( indices=-2, axis=axis )
            logy_m1     = logy.take( indices=-1, axis=axis )


            # We normally expect numerical underflow issues, not overflow
            final3logy = np.array([np.abs(logy_m1), np.abs(logy_m2), np.abs(logy_m3)])
            final3_noninf = final3logy[np.where(np.logical_not(np.isinf(final3logy)))]

            try:
                max_val = np.max(final3_noninf)/3
            except:
                # If they are all -inf then max_val doesn't matter
                max_val = 0


            addon = _alpha*np.exp(logy_m1+max_val) + _beta*np.exp(logy_m2+max_val) + -_eta*np.exp(logy_m3+max_val)

            if np.sum(addon<0)>0:
                result= np.log(np.exp(result+max_val) + addon) - max_val
            else:
                result= np.logaddexp(result, np.log(addon)-max_val)
        return result-np.log(6)
    else:
        # In the case where there are only three or less values
        return logspace_trapz(logy, x, axis=axis)


def iterate_logspace_simps(logy, axes, axisindices=None):
    logintegrandvalues = logy
            
    if axisindices is None:
        for axis in axes:
            logintegrandvalues = logspace_simpson(logy = logintegrandvalues, x=axis, axis=0)
    else:
        axisindices = np.asarray(axisindices)

        # TODO: Remove this for loop, any for loop, all the for loops
        for loop_idx, (axis, axis_idx) in enumerate(zip(axes, axisindices)):
            # Assuming the indices are in order we subtract the loop idx from the axis index
            logintegrandvalues = logspace_simpson(logy = logintegrandvalues, x=axis, axis=axis_idx-loop_idx)


    return logintegrandvalues