import numpy as np
from scipy import special




def construct_log_dx(axis: np.ndarray) -> np.ndarray|float:
    dx = np.diff(axis)
    
    # Isclose calculate with ``absolute(a - b) <= (atol + rtol * absolute(b))''
    # With the default values being atol=1e-8 and rtol=1e-5. Need to be careful
    # If we ever have differences ~1e-8 
    if np.isclose(dx[1], dx[0]):
        dx = np.append(dx, dx[-1])
    else:
        dx = axis*(10**(np.log10(axis[1])-np.log10(axis[0])))

    return np.log(dx)

def construct_log_dx_mesh(axes: list[np.ndarray] | tuple[np.ndarray]) -> np.ndarray | float:
    dxlist = []
    for axis in axes:
        dxlist.append(construct_log_dx(axis))

    logdx = np.sum(np.meshgrid(*dxlist, indexing='ij'), axis=0)

    return logdx


def logspace_riemann(logy: np.ndarray, x: np.ndarray, axis: int=-1) -> np.ndarray|float:
    """Basic 'integration' used over uniform 
    (uniform or log-uniform) discrete values"""
    logdx = construct_log_dx(x)
    indices = list(range(logy.ndim))
    indices.pop(axis)

    logdx = np.expand_dims(logdx, axis=indices)

    return special.logsumexp(logy+logdx, axis=axis)



def logspace_trapz(logy: np.ndarray, x: np.ndarray, axis: int =-1) -> np.ndarray|float:
    h = np.diff(x)

    logfkm1    = logy[:-1]
    logfk      = logy[1:]

    trapz_int = special.logsumexp(np.logaddexp(logfkm1, logfk)+np.log(h) - np.log(2), axis=axis)

    return trapz_int


def logspace_simpson(logy: np.ndarray, x: np.ndarray, axis: int =-1) -> np.ndarray|float:
    """
    Perform vectorized Simpson integration for log integrand values. 
        
    Not suitable for potentially numerically unstable x values.

    Parameters:
    - logy: Array of function values.
    - x: Array of x values (evenly spaced in log10 space).
    - logjacob: A variable to keep a standard input of all the integrators

    Returns:
    - result: Simpson's rule integration result.
    """
    if logy.shape[axis] != len(x):
        raise ValueError("Input arrays y and x must have the same length.")
    
    logy = logy

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


def iterate_logspace_integration(logy: np.ndarray, axes: np.ndarray, logspace_integrator=logspace_riemann, axisindices: list=None) -> np.ndarray|float:
    logintegrandvalues = logy
            
    if axisindices is None:
        for axis in axes:
            logintegrandvalues = logspace_integrator(logy = logintegrandvalues, x=axis, axis=0)
    else:
        axisindices = np.asarray(axisindices)

        # TODO: Remove this for loop, any for loop, all the for loops
        for loop_idx, (axis, axis_idx) in enumerate(zip(axes, axisindices)):

            # Assuming the indices are in order we subtract the loop idx from the axis index
            # print(loop_idx, axis_idx-loop_idx, axis.shape, logintegrandvalues.shape)
            try:
                logintegrandvalues = logspace_integrator(logy = logintegrandvalues, x=axis, axis=axis_idx-loop_idx)
            except:
                print(loop_idx, axis.shape, axis_idx, logintegrandvalues.shape)
                raise Exception("Error occurred during integration.")



    return logintegrandvalues