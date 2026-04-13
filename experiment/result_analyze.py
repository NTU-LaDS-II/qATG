import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def srate(counts, nqb = 1):
    return counts['0' * nqb] / sum(counts.values())
def fit_rb(result_counts, nClif, plot = False, nqb = 1):
    #nClif = np.array(lengths * num_samples)
    d = 2 ** nqb
    p0 = np.array([srate(counts, nqb) for counts in result_counts])
    print(p0, nClif)
    def fit_function(x_values, y_values, function, init_params):
        fitparams, conv = curve_fit(function, x_values, y_values, init_params)
        y_fit = function(x_values, *fitparams)

        return fitparams, y_fit, conv
    try:
        fit_params, y_fit, conv = fit_function(nClif, p0, 
                    lambda x, a, b, alpha: a * alpha ** x + b,
                    [1 - 1 / d, 1 / d, 0.999]
                    )
    except:
        return float('nan'), float('nan')
    print(fit_params)
    c = [np.sqrt(conv[i][i]) for i in range(len(conv))]
    a, b, alpha = fit_params
    EPC = (d - 1) / d * (1 - alpha)
    err = c[2] / 2
    if (plot):
        plt.scatter(nClif, p0, color='black')
        plt.plot(nClif, y_fit, color='red', label=f"alpha = {alpha:.5f}+/-{c[2]}")
        #plt.errorbar(times_us, y_fit, yerr=c, fmt="o")
        #plt.xlim(0, np.max(times_us))
        plt.title("RB Experiment", fontsize=15)
        plt.xlabel('Number of Clifford', fontsize=15)
        plt.ylabel('P(0)', fontsize=15)
        plt.legend()
        plt.show()
    print("EPC:", EPC)
    return EPC, err