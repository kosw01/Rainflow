"""
Rainflow Cycle Counting Algorithm for Python
Translated from MATLAB code by Adam Nieslony
ASTM E 1049-85 standard implementation

Author: Translated to Python
Original: Adam Nieslony (ajn@po.opole.pl)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Union


def sig2ext(sig: np.ndarray, dt: Union[float, np.ndarray] = 1.0, 
            clsn: Optional[int] = None, plot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    SIG2EXT - searches local extrema from time course (signal)
    
    Parameters
    ----------
    sig : array_like
        Time course of loading signal
    dt : float or array_like, optional
        Sampling time (default: 1.0) or time vector
    clsn : int, optional
        Number of classes for signal quantization before extrema search
    plot : bool, optional
        If True, plot the signal with extrema marked (default: False)
    
    Returns
    -------
    ext : ndarray
        Found extrema from signal
    exttime : ndarray
        Time of extremum occurrence
    """
    sig = np.asarray(sig).flatten()
    
    # Handle dt parameter
    if isinstance(dt, (int, float)):
        dt = float(dt)
        dt_is_scalar = True
    else:
        dt = np.asarray(dt).flatten()
        dt_is_scalar = False
        if len(dt) != len(sig):
            raise ValueError("dt must have same length as sig or be a scalar")
    
    # Store original signal for plotting
    if plot or clsn is not None:
        oldsig = sig.copy()
    
    # Quantize signal if clsn is provided
    if clsn is not None:
        clsn = int(clsn) - 1
        smax = np.max(sig)
        smin = np.min(sig)
        if smax != smin:
            sig = np.round((sig - smin) * clsn / (smax - smin)) * (smax - smin) / clsn + smin
    
    # Find extrema: binary vector where 1 means extremum
    # First and last points are considered extrema
    w1 = np.diff(sig)
    w = np.zeros(len(sig), dtype=bool)
    w[0] = True
    w[-1] = True
    w[1:-1] = (w1[:-1] * w1[1:]) <= 0
    
    ext = sig[w]
    
    # Calculate time for extrema
    if dt_is_scalar:
        exttime = (np.where(w)[0]) * dt
    else:
        exttime = dt[w]
    
    # Remove triple values (three consecutive equal values)
    if len(ext) > 2:
        w1 = np.diff(ext)
        w_mask = np.ones(len(ext), dtype=bool)
        w_mask[1:-1] = ~((w1[:-1] == 0) & (w1[1:] == 0))
        ext = ext[w_mask]
        exttime = exttime[w_mask]
    
    # Remove double values and shift time to center
    if len(ext) > 1:
        w_mask = np.ones(len(ext), dtype=bool)
        w_mask[1:] = ext[:-1] != ext[1:]
        ext = ext[w_mask]
        
        # Shift time to center for removed points
        if len(exttime) > 1:
            w1 = np.diff(exttime) / 2
            exttime_new = exttime[:-1].copy()
            exttime_new[1:] = exttime_new[1:] + w1[:-1] * (~w_mask[1:-1])
            exttime = np.concatenate([exttime_new, [exttime[-1]]])
            exttime = exttime[w_mask]
    
    # Final check for extrema
    if len(ext) > 2:
        w1 = np.diff(ext)
        w_mask = np.ones(len(ext), dtype=bool)
        w_mask[1:-1] = (w1[:-1] * w1[1:]) < 0
        ext = ext[w_mask]
        exttime = exttime[w_mask]
    
    # Plot if requested
    if plot:
        if dt_is_scalar:
            time_vec = np.arange(len(oldsig)) * dt
        else:
            time_vec = dt
        
        plt.figure(figsize=(10, 6))
        if clsn is not None:
            plt.plot(time_vec, oldsig, 'b-', label='SIGNAL', linewidth=1.5)
            plt.plot(time_vec, sig, 'g-', label='SIGNAL + CLS', linewidth=1.5)
        else:
            plt.plot(time_vec, sig, 'b-', label='SIGNAL', linewidth=1.5)
        plt.plot(exttime, ext, 'ro', label='EXTREMA', markersize=8)
        plt.xlabel('time')
        plt.ylabel('signal & extrema')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Signal with Extrema')
        plt.tight_layout()
        plt.show()
    
    return ext, exttime


def rainflow(ext: np.ndarray, dt: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """
    RAINFLOW cycle counting
    
    Parameters
    ----------
    ext : array_like
        Signal points (turning points/extrema)
    dt : float or array_like, optional
        Sampling time (scalar) or time vector for each point
        
    Returns
    -------
    rf : ndarray
        Rainflow cycles matrix:
        - rf[0, :] : Cycles amplitude
        - rf[1, :] : Cycles mean value
        - rf[2, :] : Number of cycles (0.5 or 1.0)
        - rf[3, :] : Beginning time (if dt provided)
        - rf[4, :] : Cycle period (if dt provided)
        
        Shape: (3, n) if dt is None, (5, n) if dt is provided
    """
    ext = np.asarray(ext).flatten()
    tot_num = len(ext)
    
    # Handle time parameter
    if dt is None:
        return _rf3(ext)
    else:
        if isinstance(dt, (int, float)):
            # Create time vector from scalar dt
            extt = np.arange(tot_num) * float(dt)
        else:
            extt = np.asarray(dt).flatten()
            if len(extt) != tot_num:
                raise ValueError("Time array must have same length as extrema array")
        return _rf5(ext, extt)


def _rf3(ext: np.ndarray) -> np.ndarray:
    """
    Rainflow without time analysis
    Returns: [amplitude, mean, cycle_count] x n
    """
    tot_num = len(ext)
    a = np.zeros(512)  # Stack array
    results = []
    
    j = -1
    cNr = 1
    
    for index in range(tot_num):
        j += 1
        a[j] = ext[index]
        
        while j >= 2 and abs(a[j-1] - a[j-2]) <= abs(a[j] - a[j-1]):
            ampl = abs((a[j-1] - a[j-2]) / 2)
            
            if j == 2:
                mean = (a[0] + a[1]) / 2
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if ampl > 0:
                    results.append([ampl, mean, 0.50])
            else:
                mean = (a[j-1] + a[j-2]) / 2
                a[j-2] = a[j]
                j = j - 2
                if ampl > 0:
                    results.append([ampl, mean, 1.00])
                    cNr += 1
    
    # Process remaining points
    for index in range(j):
        ampl = abs(a[index] - a[index+1]) / 2
        mean = (a[index] + a[index+1]) / 2
        if ampl > 0:
            results.append([ampl, mean, 0.50])
    
    if len(results) == 0:
        return np.zeros((3, 0))
    
    rf = np.array(results).T
    return rf


def _rf5(ext: np.ndarray, extt: np.ndarray) -> np.ndarray:
    """
    Rainflow with time analysis
    Returns: [amplitude, mean, cycle_count, begin_time, period] x n
    """
    tot_num = len(ext)
    a = np.zeros(512)  # Stack array for values
    t = np.zeros(512)  # Stack array for times
    results = []
    
    j = -1
    cNr = 1
    
    for index in range(tot_num):
        j += 1
        a[j] = ext[index]
        t[j] = extt[index]
        
        while j >= 2 and abs(a[j-1] - a[j-2]) <= abs(a[j] - a[j-1]):
            ampl = abs((a[j-1] - a[j-2]) / 2)
            
            if j == 2:
                mean = (a[0] + a[1]) / 2
                period = (t[1] - t[0]) * 2
                atime = t[0]
                a[0] = a[1]
                a[1] = a[2]
                t[0] = t[1]
                t[1] = t[2]
                j = 1
                if ampl > 0:
                    results.append([ampl, mean, 0.50, atime, period])
            else:
                mean = (a[j-1] + a[j-2]) / 2
                period = (t[j-1] - t[j-2]) * 2
                atime = t[j-2]
                a[j-2] = a[j]
                t[j-2] = t[j]
                j = j - 2
                if ampl > 0:
                    results.append([ampl, mean, 1.00, atime, period])
                    cNr += 1
    
    # Process remaining points
    for index in range(j):
        ampl = abs(a[index] - a[index+1]) / 2
        mean = (a[index] + a[index+1]) / 2
        period = (t[index+1] - t[index]) * 2
        atime = t[index]
        if ampl > 0:
            results.append([ampl, mean, 0.50, atime, period])
    
    if len(results) == 0:
        return np.zeros((5, 0))
    
    rf = np.array(results).T
    return rf


def rfhist(rf: np.ndarray, x: Union[int, np.ndarray] = 10, 
           rfflag: str = 'ampl', plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram for rainflow data
    
    Parameters
    ----------
    rf : ndarray
        Rainflow data from rainflow() function
    x : int or array_like
        Number of bins (if int) or bin centers (if array)
    rfflag : str
        Data type flag: 'ampl', 'mean', 'freq', or 'period'
    plot : bool
        If True, plot the histogram (default: True)
    
    Returns
    -------
    no : ndarray
        Number of extracted cycles
    xo : ndarray
        Bin locations
    """
    if rfflag[0].lower() == 'm':
        r = 1  # mean (row 2, index 1)
        xl = 'Histogram of "rainflow" cycles mean value'
    elif rfflag[0].lower() == 'f':
        r = 4  # frequency (row 5, index 4)
        rf = rf.copy()
        rf[4, :] = rf[4, :] ** -1  # Convert period to frequency
        xl = 'Histogram of "rainflow" cycles frequency'
    elif rfflag[0].lower() == 'p':
        r = 4  # period (row 5, index 4)
        xl = 'Histogram of "rainflow" cycles period'
    else:
        r = 0  # amplitude (row 1, index 0)
        xl = 'Histogram of "rainflow" amplitudes'
    
    # Find half-cycles
    halfc = np.where(rf[2, :] == 0.5)[0]
    
    # Calculate histogram for all data
    if isinstance(x, int):
        N1, xo = np.histogram(rf[r, :], bins=x)
        xo = (xo[:-1] + xo[1:]) / 2  # Convert edges to centers
    else:
        x = np.asarray(x).flatten()
        N1, xo = np.histogram(rf[r, :], bins=x)
        xo = x
    
    # Adjust for half-cycles
    if len(halfc) > 0:
        if isinstance(x, int):
            N2, _ = np.histogram(rf[r, halfc], bins=x)
        else:
            N2, _ = np.histogram(rf[r, halfc], bins=x)
        N1 = N1 - 0.5 * N2
    else:
        N2 = np.zeros_like(N1)
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(xo, N1, width=(xo[-1] - xo[0]) / len(xo) * 0.8, align='center')
        plt.xlabel(xl)
        plt.ylabel(f'Nr of cycles: {np.sum(N1):.1f} ({np.sum(N2)/2:.1f} from half-cycles)')
        plt.title('Rainflow Histogram')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    return N1, xo


def rfmatrix(rf: np.ndarray, x: int = 10, y: int = 10,
             flagx: str = 'ampl', flagy: str = 'mean', plot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rainflow matrix estimation
    
    Parameters
    ----------
    rf : ndarray
        Rainflow data from rainflow() function
    x, y : int or array_like
        Number of bins in x and y direction (if int) or bin centers (if array)
    flagx, flagy : str
        Data type flags: 'ampl', 'mean', 'freq', or 'period'
    plot : bool
        If True, plot the 3D matrix (default: True)
    
    Returns
    -------
    m : ndarray
        Rainflow matrix
    mx : ndarray
        Bin locations for x direction
    my : ndarray
        Bin locations for y direction
    """
    # Extract data based on flags
    if flagx[0].lower() == 'm':
        xdata = rf[1, :]  # mean
    elif flagx[0].lower() == 'f':
        xdata = rf[4, :] ** -1  # frequency
    elif flagx[0].lower() == 'p':
        xdata = rf[4, :]  # period
    else:
        xdata = rf[0, :]  # amplitude
    
    if flagy[0].lower() == 'm':
        ydata = rf[1, :]  # mean
    elif flagy[0].lower() == 'f':
        ydata = rf[4, :] ** -1  # frequency
    elif flagy[0].lower() == 'p':
        ydata = rf[4, :]  # period
    else:
        ydata = rf[0, :]  # amplitude
    
    cdata = rf[2, :]  # cycle count
    
    # Process x bins
    if isinstance(x, int):
        minx = np.min(xdata)
        maxx = np.max(xdata)
        binwidth = (maxx - minx) / x
        xx_edges = np.linspace(minx, maxx, x + 1)
        xx = (xx_edges[:-1] + xx_edges[1:]) / 2  # Centers
    else:
        x = np.asarray(x).flatten()
        xx = x
        binwidth = np.diff(xx)
        if len(binwidth) > 0:
            binwidth = np.concatenate([[binwidth[0]], binwidth, [binwidth[-1]]])
        else:
            binwidth = np.array([1.0])
        xx_edges = np.concatenate([[xx[0] - binwidth[0]/2], xx + binwidth[1:-1]/2, [xx[-1] + binwidth[-1]/2]])
    
    # Process y bins
    if isinstance(y, int):
        miny = np.min(ydata)
        maxy = np.max(ydata)
        binwidth = (maxy - miny) / y
        yy_edges = np.linspace(miny, maxy, y + 1)
        yy = (yy_edges[:-1] + yy_edges[1:]) / 2  # Centers
    else:
        y = np.asarray(y).flatten()
        yy = y
        binwidth = np.diff(yy)
        if len(binwidth) > 0:
            binwidth = np.concatenate([[binwidth[0]], binwidth, [binwidth[-1]]])
        else:
            binwidth = np.array([1.0])
        yy_edges = np.concatenate([[yy[0] - binwidth[0]/2], yy + binwidth[1:-1]/2, [yy[-1] + binwidth[-1]/2]])
    
    # Build matrix
    srf = np.zeros((len(yy), len(xx)))
    
    # Process first y bin
    rfk = np.where((ydata >= yy_edges[0]) & (ydata <= yy_edges[1]))[0]
    if len(rfk) > 0:
        rf_sub = np.vstack([xdata[rfk], ydata[rfk], cdata[rfk]])
        hist, _ = rfhist(rf_sub, xx, 'ampl', plot=False)
        srf[0, :] = hist
    
    # Process remaining y bins
    for k in range(1, len(yy)):
        rfk = np.where((ydata > yy_edges[k]) & (ydata <= yy_edges[k+1]))[0]
        if len(rfk) > 0:
            rf_sub = np.vstack([xdata[rfk], ydata[rfk], cdata[rfk]])
            hist, _ = rfhist(rf_sub, xx, 'ampl', plot=False)
            srf[k, :] = hist
    
    # Plot if requested
    if plot:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plot
        X, Y = np.meshgrid(xx, yy)
        
        # Create bar plot
        dx = (xx[-1] - xx[0]) / len(xx) * 0.8 if len(xx) > 1 else 1
        dy = (yy[-1] - yy[0]) / len(yy) * 0.8 if len(yy) > 1 else 1
        
        xpos = X.flatten()
        ypos = Y.flatten()
        zpos = np.zeros_like(xpos)
        dx_arr = np.full_like(xpos, dx)
        dy_arr = np.full_like(ypos, dy)
        dz = srf.flatten()
        
        # Plot bars
        colors = cm.viridis(dz / (dz.max() + 1e-10))
        ax.bar3d(xpos, ypos, zpos, dx_arr, dy_arr, dz, color=colors, alpha=0.8)
        
        ax.set_xlabel(f'X - {flagx}')
        ax.set_ylabel(f'Y - {flagy}')
        ax.set_zlabel('number of cycles')
        ax.set_title('Rain Flow Matrix')
        plt.tight_layout()
        plt.show()
    
    return srf, xx, yy


def rfdemo1(ext: Optional[Union[int, np.ndarray]] = None):
    """
    Demo showing cycles extracted from signal using rainflow algorithm
    Recommended for short signals
    
    Parameters
    ----------
    ext : int or array_like, optional
        If int: number of random points to generate
        If array: signal to process
        If None: use 16 random points
    """
    if ext is None:
        ext = sig2ext(np.random.randn(4), plot=False)[0]
    elif isinstance(ext, (int, np.int_)):
        ext = sig2ext(np.random.randn(ext), plot=False)[0]
    else:
        ext = sig2ext(ext, plot=False)[0]
    
    a = rainflow(ext, 1.0)
    m, n = a.shape
    
    if n > 100:
        response = input(f'Rainflow found {np.sum(a[2, :]):.1f} cycles! Do you want to continue? (y/n): ')
        if response.lower() != 'y':
            print('Function aborted by user.')
            return
    
    col = ['y', 'm', 'c', 'r', 'g', 'b']
    
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(ext)), ext, 'k.:', markersize=10, label='peaks from signal')
    
    wyk = np.arange(0, 1.05, 0.05)
    
    for c in range(n):
        colnr = c % 6
        
        nr1 = int(round(a[3, c]))
        nr2 = int(round(a[3, c] + a[4, c] * a[2, c]))
        
        if a[2, c] == 1.0:
            if nr1 < len(ext) - 1 and ext[nr1] < ext[nr1 + 1]:
                t_cycle = wyk * a[4, c] + a[3, c]
                y_cycle = np.cos(np.pi + wyk * 2 * np.pi) * a[0, c] + a[1, c]
                plt.plot(t_cycle, y_cycle, color=col[colnr], linewidth=2)
                plt.text(a[3, c], a[1, c] - a[0, c], f'{c+1}. Cycle, up',
                        color=col[colnr], verticalalignment='top')
            else:
                t_cycle = wyk * a[4, c] + a[3, c]
                y_cycle = np.cos(wyk * 2 * np.pi) * a[0, c] + a[1, c]
                plt.plot(t_cycle, y_cycle, color=col[colnr], linewidth=2)
                plt.text(a[3, c], a[1, c] + a[0, c], f'{c+1}. Cycle, down',
                        color=col[colnr], verticalalignment='bottom')
        else:
            if nr1 < len(ext) and nr2 < len(ext) and ext[nr1] > ext[nr2]:
                t_cycle = wyk * a[4, c] * 0.5 + a[3, c]
                y_cycle = np.cos(wyk * np.pi) * a[0, c] + a[1, c]
                plt.plot(t_cycle, y_cycle, color=col[colnr], linewidth=2)
                plt.text(a[3, c], a[1, c] + a[0, c], f'{c+1}. Half-cycle, down',
                        color=col[colnr], verticalalignment='bottom')
            else:
                t_cycle = wyk * a[4, c] * 0.5 + a[3, c]
                y_cycle = np.cos(np.pi + wyk * np.pi) * a[0, c] + a[1, c]
                plt.plot(t_cycle, y_cycle, color=col[colnr], linewidth=2)
                plt.text(a[3, c], a[1, c] - a[0, c], f'{c+1}. Half-cycle, up',
                        color=col[colnr], verticalalignment='top')
    
    plt.xlabel('peaks, counted from 0')
    plt.ylabel('value')
    plt.title('Rainflow cycles extracted from signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print('Row 1: amplitude')
    print('Row 2: mean')
    print('Row 3: number of cycles (cycle or half cycle)')
    print('Row 4: begin time of extracted cycle or half cycle')
    print('Row 5: period of a cycle')
    print(a)


def rfdemo2(ext: Optional[Union[int, np.ndarray]] = None):
    """
    Demo for rainflow matrix and histograms
    Recommended for long signals (10000+ points)
    
    Parameters
    ----------
    ext : int or array_like, optional
        If int: number of random points to generate
        If array: signal to process
        If None: use 10000 random points
    """
    if ext is None:
        ext = sig2ext(np.random.randn(10000), plot=False)[0]
    elif isinstance(ext, (int, np.int_)):
        ext = sig2ext(np.random.randn(ext), plot=False)[0]
    else:
        ext = sig2ext(ext, plot=False)[0]
    
    ext = sig2ext(ext, plot=False)[0]  # Ensure extrema
    rf = rainflow(ext)
    
    rfhist(rf, 30, 'ampl')
    rfhist(rf, 30, 'mean')
    rfmatrix(rf, 30, 30)


# Example usage
if __name__ == "__main__":
    print("Rainflow Cycle Counting - Python Implementation")
    print("=" * 50)
    
    # Example 1: Simple signal
    print("\nExample 1: Simple signal analysis")
    signal = 10 * np.random.randn(1000) + np.random.rand(1000)
    ext, exttime = sig2ext(signal, dt=0.01, plot=True)
    
    # Example 2: Rainflow counting
    print("\nExample 2: Rainflow counting")
    rf = rainflow(ext, exttime)
    print(f"Found {rf.shape[1]} cycles")
    print(f"Total cycles: {np.sum(rf[2, :]):.2f}")
    
    # Example 3: Histogram
    print("\nExample 3: Histogram")
    rfhist(rf, 20, 'ampl')
    
    # Example 4: Matrix
    print("\nExample 4: Rainflow matrix")
    rfmatrix(rf, 15, 15)
    
    print("\nDone!")

