import numpy as np
from matplotlib import pyplot as plt

def simulate_series(t0, w, length=2000, lamb = 0.3):
    t = np.expand_dims(np.arange(length), 0)
    return np.sin((t - np.expand_dims(t0, 1))/np.expand_dims(w, 1)) + lamb * np.random.normal(size=length)

def overlaps(start, prev_starts):
    for prev_start in prev_starts:
        if prev_start > start - 90 and prev_start < start + 90:
            return True
    return False

def inject_shockwaves(series, num_series=3, num_shockwaves=5, durations=(10,30,90), shock_range=(0.7, 0.9), verbose=False):
    shocked_i = np.random.randint(series.shape[0], size=num_series)
    shocked_series = series[shocked_i]
    series_starts = [[]] * len(shocked_series)
    for i_series, duration in enumerate([durations[i % len(durations)] for i in range(num_shockwaves)]):
        siri = shocked_series[i_series % len(shocked_series)]
        prev_starts = series_starts[i_series % len(shocked_series)]

        start = np.random.randint(len(siri) - duration)
        while overlaps(start, prev_starts):
            start = np.random.randint(len(siri) - duration)
        if verbose:
            print(i_series % len(shocked_series), start)
        series[shocked_i[i_series % len(shocked_series)], start:start+duration] += \
            np.random.uniform(shock_range[0], shock_range[1], size=duration) * np.random.choice([-1, 1])
        prev_starts.append(start)
    return shocked_i

def get_simulated():
    series = simulate_series(np.linspace(50,100,30), np.linspace(40,50,30))
    shocked_i = inject_shockwaves(series)
    np.save('simulated_pmu.npy', series)
    return series, shocked_i

if __name__ == '__main__':
    series, shocked_i = get_simulated()
    for siri in series[shocked_i]:
        plt.plot(siri)
    plt.show()
