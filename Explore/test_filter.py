
import numpy as np
import scipy
import scipy.signal

from scipy import signal
import matplotlib.pyplot as plt

from numba import jit

def explore():
    t = np.linspace(-1, 1, 201)
    x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +
         0.1*np.sin(2*np.pi*1.25*t + 1) +
         0.18*np.cos(2*np.pi*3.85*t))
    xn = x + np.random.randn(len(t)) * 0.08

    # Create an order 3 lowpass butterworth filter:
    b, a = signal.butter(10, 0.05)

    # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

    # Apply the filter again, to have a result filtered at an order the same as filtfilt:
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

    # Use filtfilt to apply the filter:
    y = signal.filtfilt(b, a, xn)

    # Plot the original signal and the various filtered versions:

    plt.figure()
    plt.plot(t, xn, 'b', alpha=0.75)
    plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
    plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
                'filtfilt'), loc='best')

    plt.grid(True)
    plt.show()

    plt.figure(3)
    plt.clf()
    y = signal.filtfilt(b, a, deltas)
    mm = signal.medfilt(deltas, 31)
    plt.plot(deltas, 'b', alpha=0.75)
    plt.plot(y, 'r--')
    plt.plot(mm, 'g--')
    plt.grid(True)
    plt.show()


def plot_filter(b, a, title="filter"):
    w, h = scipy.signal.freqz(b, a)
    plt.figure()
    plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    plt.title(title)
    plt.xlabel('Normalized frequency')
    plt.ylabel('Amplitude [dB]')
    plt.grid(which='both', axis='both')
    plt.fill([.01, 0.2, 0.2, .01], [-3, -3, -99, -99], '0.9', lw=0)  # stop
    plt.fill([0.3, 0.3,   2,   2], [ 9, -40, -40,  9], '0.9', lw=0)  # pass
    plt.axis([0.08, 1, -60, 3])
    plt.show()


def local_average_filter(width):
    "Super simple running average."
    b = np.ones((width)) / width
    a = np.zeros((width))
    a[0] = 1
    return b, a


def butterworth_filter(cutoff, order, sampling_frequency=10):
    fs = sampling_frequency     # Sampling frequency
    omega = cutoff / (fs / 2)   # Normalize the frequency
    return scipy.signal.butter(order, omega, 'low')


def chebyshev_filter(cutoff, order, sampling_frequency=10):
    fs = sampling_frequency     # Sampling frequency
    omega = cutoff / (fs / 2)   # Normalize the frequency
    N, Wn = scipy.signal.cheb1ord(omega, 1.5 * omega, 3, 40)
    b, a = scipy.signal.cheby1(order, 1, Wn, 'low')
    return b, a

def exponential_filter(cutoff):
    b, a = butterworth_filter(cutoff, 1)
    alpha = 1 - a.sum()
    b = np.array([(1 - alpha), 0.0])
    a = np.array([1.0, -alpha])
    return b, a

@jit(nopython=True)
def exponential_nonlinear_filter_helper(sig, alpha, max_error):
    beta = alpha
    res = sig.copy()
    betas = np.zeros(sig.shape)
    betas[0] = beta
    for i in range(2, len(sig)):
        res[i] = beta/2 * (sig[i] + sig[i-1]) + (1 - beta) * res[i-1]
        if np.abs(res[i] - sig[i]) > max_error:
            beta = min(1.5 * beta, 0.4)
        else:
            beta = (beta + alpha)/2
        betas[i] = beta
    return res, betas

def exponential_nonlinear_filter(sig, cutoff, max_error):
    b, a = butterworth_filter(cutoff, 1)
    alpha = b.sum()
    return exponential_nonlinear_filter_helper(sig, alpha, max_error)


def test_filter():
    plt.clf()
    fs = 10
    fc = 0.3
    omega = fc / (fs / 2)
    print(f"Omega is {omega}")
    legend = []
    for order in [1, 2, 3, 5]:
        b, a = signal.butter(order, omega, 'low')
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        legend.append(f'Order {order}')

    for multiplier in [2, 3,  4]:
        b, a = exponential_filter(multiplier * omega)
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        legend.append(f'Exponential {multiplier}')

    b, a = exponential_filter(2 * omega)
    w, h = signal.freqz(b, a)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    legend.append(f'Exponential')

    
    if False:
        N, Wn = signal.cheb1ord(omega, 1.5 * omega, 3, 40)
        N = 7
        b, a = signal.cheby1(N, 1, Wn, 'low')
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        legend.append(f'Cheby')
    plt.legend(legend)


def test_smooth_helper(signal, fignum=1):
    """
    Test various settings of butterworth and chebyshev on a step edges.  Both have
    noticeable ringing, but butter is more accurate.  Lower order butter look more
    responsive in part because they let high frequencies pass.
    """

    plt.figure(fignum)
    plt.clf()
    plt.plot(signal, linestyle = 'None', marker='.')

    fs = 10                     # Sampling frequency
    fc = 0.3                    # 0.3 Hz, Cut-off frequency of the filter
    omega = fc / (fs / 2)       # Normalize the frequency

    legend_list = ['signal']
    for order in [9]: # [3, 5, 7, 9]:
        b, a = scipy.signal.butter(order, omega, 'low')
        # Use lfilter to apply the filter:
        bfiltered = scipy.signal.lfilter(b, a, signal)
        legend_list.append(f'order {order}')
        plt.plot(bfiltered)

    b, a = scipy.signal.butter(1, omega, 'low')
    print(b, a)
    alpha = 1 - a.sum()
    b, a = exponential_filter(alpha)
    print(b, a)
    bfiltered = scipy.signal.lfilter(b, a, signal)
    plt.plot(bfiltered)
    legend_list.append(f'exponential')

    efiltered, _ = exponential_nonlinear_filter(signal, fc, 4)
    plt.plot(efiltered)
    legend_list.append(f'nonlinexp')

    N, Wn = scipy.signal.cheb1ord(omega, 1.5 * omega, 3, 40)
    N = 5
    b, a = scipy.signal.cheby1(N, 1, Wn, 'low')

    cfiltered = scipy.signal.lfilter(b, a, signal)
    plt.plot(cfiltered)
    legend_list.append('chebyshev'.format(order))

    plt.legend(legend_list, loc='best')
    plt.grid(True)
    plt.show()
    

def test_smooth():
    sig1 = 40 * np.hstack([np.zeros(1000), np.ones(1000)])
    test_smooth_helper(sig1, 1)
    sig2 = sig1 + 40 * 0.03 * np.random.randn(len(sig1))
    test_smooth_helper(sig2, 2)
    
    plt.clf()
    sig1 = 40 * np.hstack([np.zeros(1000), np.ones(1000)])
    sig2 = sig1 + 40 * 0.03 * np.random.randn(len(sig1))
    efiltered, betas = exponential_nonlinear_filter(sig2, 0.02, 4)

    plt.plot(sig2)
    plt.plot(efiltered)
    plt.plot(betas*10)
