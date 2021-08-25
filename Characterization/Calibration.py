#QUESTO FUNZIONA SUGLI JUPYTER NOTEBOOK
#Dal sito: https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html
from qiskit import *
from qiskit.tools.jupyter import *
import tensorflow as tf
from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')

backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"

dt = backend_config.dt
print(f"Sampling time: {dt*1e9} ns")    # The configuration returns dt in seconds, so multiply by
                                        # 1e9 to get nanoseconds

backend_defaults = backend.defaults()

import numpy as np

GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

qubit = 0   # qubit che studiamo
mem_slot = 0    # Memory slot

# Centro della finestra di sweep. Frequenza espressa in Hertz
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]

print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

# Fattore di scala (per comodità)
scale_factor = 1e-14

# Ampiezza finestra di sweep
frequency_span_Hz = 40 * MHz
# step di cui mi muovo all'interno della finestra
frequency_step_Hz = 1 * MHz

frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2

# Vettore np delle freq. dell'esperimento. Definisco primo, ultimo numero del vettore e la spaziatura tra i valori
frequencies_GHz = np.arange(frequency_min / GHz, frequency_max / GHz, frequency_step_Hz / GHz)

print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
in steps of {frequency_step_Hz / MHz} MHz.")

# Siccome i samples devono essere in numero multiplo di 16, definisco una funzione
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)


from qiskit import pulse                  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter      # This is Parameter Class for variable parameters.

# Drive pulse parameters (us = microseconds)
drive_sigma_sec = 0.075 * us                            # This determines the actual width of the gaussian
drive_duration_sec = drive_sigma_sec * 8                # This is a truncating parameter, because gaussians don't have
                                                            # a natural finite length
drive_amp = 0.05

# Creare il programma di base
# Start with drive pulse acting on the drive channel
freq = Parameter('freq')
with pulse.build(backend=backend, default_alignment='sequential', name='Frequency sweep') as sweep_sched:
    drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
    drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(freq, drive_chan)
    # Drive pulse samples
    pulse.play(pulse.Gaussian(duration=drive_duration,
                              sigma=drive_sigma,
                              amp=drive_amp,
                              name='freq_sweep_excitation_pulse'), drive_chan)
    # Define our measurement pulse
    pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])


# Create the frequency settings for the sweep (MUST BE IN HZ)
frequencies_Hz = frequencies_GHz*GHz

schedules = [sweep_sched.assign_parameters({freq : f}, inplace=False) for f in frequencies_Hz]

schedules[0].draw() #Argomento backend=backend

num_shots_per_frequency = 1024

job = backend.run(schedules,
                  meas_level=1,
                  meas_return='avg',
                  shots=num_shots_per_frequency)

# Monitoro lo stato del job
from qiskit.tools.monitor import job_monitor
job_monitor(job)

# Recupero i risultati
frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds

# Estraggo i risultati e li plotto
import matplotlib.pyplot as plt

sweep_values = []
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    res = frequency_sweep_results.get_memory(i)*scale_factor
    # Get the results for `qubit` from this experiment
    sweep_values.append(res[qubit])

plt.scatter(frequencies_GHz, np.real(sweep_values), color='black') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured signal [a.u.]")
plt.show()

from scipy.optimize import curve_fit

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit

fit_params, y_fit = fit_function(frequencies_GHz,
                                  np.real(sweep_values),
                                  lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                  [5, 4.975, 1, -15] # initial parameters for curve_fit
                                )

plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
plt.plot(frequencies_GHz, y_fit, color='red')
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])

plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Signal [a.u.]")
plt.show()

A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print(f"We've updated our qubit frequency estimate from "
      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")
