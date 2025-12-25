import os
import numpy as np
import sdeint
from tqdm import tqdm

from timkf.core import NumpyTwoComponentPhase
from timkf.constants import DAY_TO_SECONDS

def simulate_toas_random(injection_psr_level, Nobs, Tdays, 
                         output_parfile, output_timfile, PEPOCH=0, seed=None):
    """
    Function directly copied from https://github.com/oneill-academic/pulsar_freq_filter/blob/main/pulsar_freq_filter/toa_simulations/test01_random_toas/programs/simulate_toas_random.py
    """
    rng = np.random.default_rng(seed)
    sigmac = np.sqrt(injection_psr_level["Qc"])
    sigmas = np.sqrt(injection_psr_level["Qs"])
    omgc_0 = injection_psr_level["omgc_0"]
    omgc_dot = injection_psr_level["omgc_dot"]
    phi_0 = injection_psr_level["phic_0"]
    lag = injection_psr_level["lag"]
    T_error_in = np.sqrt(injection_psr_level["Rc"]) / omgc_0

    print('Simulate nonuniform TOAs from nothing')
    mytimes = np.linspace(PEPOCH, Tdays, Nobs*1000) # ED: changed to use PEPOCH instead of 0
    indexs = np.sort(np.random.choice(Nobs*1000, Nobs, replace = False)) # ED: changed to use rng
    mytimes = mytimes[indexs]
    pets0 = mytimes[0]
    tstarts = mytimes*DAY_TO_SECONDS
    toa_errors = np.ones(Nobs) * T_error_in
    omgc_0 = omgc_0 + omgc_dot*(pets0 - PEPOCH)*DAY_TO_SECONDS

    #Write a new parfile with the frequency at pets0 instead of PEPOCH
    write_par(output_parfile, omgc_0/(2*np.pi), omgc_dot/(2*np.pi), pets0)

    tauc, taus, Qc, Qs, Nc, Ns = NumpyTwoComponentPhase.param_map(injection_psr_level) # ED: changed syntax to use param_map, consistent with package
    #Set up the two component model for the simulation
    F = np.array([[0, 1/(2*np.pi), 0], [0, -1/tauc, 1/tauc], [0, 1/taus, -1/taus]])
    N = np.array([0, Nc, Ns])
    Q = np.diag([0, sigmac, sigmas])
    def f(x, t):
        return F.dot(x) + N
    def g(x, t):
        return Q
    
    toa_fracs = []
    toa_ints = []
    # phis = [] # ED: not used, suppressed
    omegacs = []
    omegass = []
    skipsize = min(1000, min(tstarts[1:]-tstarts[:-1]))
    print("skipsize =", skipsize)
    print("2*np.pi/omgc_0 =", 2*np.pi/omgc_0)

    #Simulate frequencies and phases of the pulsar to calculate TOAs
    #Set initial state
    omgs_0 = omgc_0 - lag
    p0 = np.asarray([phi_0, omgc_0, omgs_0])
    prev_tstart = tstarts[0]
    #The first time through the loop the while loop is skipped, it just evolves the system to an integer phase
    for next_tstart in tqdm(tstarts):
        #Move from prev_start to next_start
        times = [prev_tstart]
        while next_tstart > times[-1]:
            #Evolve forward to next_tstart in small steps to maintain precision
            times = np.linspace(times[-1], times[-1] + min(skipsize, next_tstart - times[-1]), num=2)
            states = sdeint.itoint(f, g, p0, times) # ED: added generator=rng
            #Reset current state
            p0 = states[-1, :]
            #Wrap phase
            p0[0] = p0[0] - np.floor(p0[0])
        #Evolve the system forward to make the phase approximately a whole number
        extra_time = np.longdouble(1 - p0[0]) * 2 * np.pi / np.longdouble(p0[1])
        newtimes = np.linspace(0, extra_time, num=2)
        states_toa = sdeint.itoint(f, g, p0, newtimes) # ED: added generator=rng
        #Add the new toas to the list
        toa = times[-1] + extra_time
        toa_fracs.append(toa - np.floor(toa))
        toa_ints.append(np.floor(toa))
        #Add the new omega_c and omega_s values to their lists
        omegacs.append(states_toa[-1, 1])
        omegass.append(states_toa[-1, 2])
        #Update for the next cycle. 
        prev_tstart = toa
        p0 = states_toa[-1, :]

    omegacs = np.asarray(omegacs)
    omegass = np.asarray(omegass)
    #Convert toas from seconds to days for TEMPO2
    toas = np.longdouble(toa_ints) / DAY_TO_SECONDS + np.longdouble(toa_fracs) / DAY_TO_SECONDS
    #Add measurement noise to TOAs
    measurement_noise = np.random.normal(0, 1, toas.size) * toa_errors / DAY_TO_SECONDS # ED: changed to use rng
    toas += measurement_noise

    #Make sure the TOAs haven't been put out of order (unlikely)
    indexs = toas.argsort()
    toas = toas[indexs]
    toa_errors = toa_errors[indexs]
    omegacs = omegacs[indexs]
    omegass = omegass[indexs]

    #Save the frequency values and times
    write_tim_file(output_timfile, toas, toa_errors)


def write_tim_file(filename, toas, toa_errors):
    # ED: added os.makedirs to create directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    #File name should end in '.tim'
    with open(filename, 'w') as myf:
        print("FORMAT 1", file=myf)
        print("MODE 1", file=myf)
        for toa, terr in zip(toas, toa_errors):
            print(f"fake 1000 {toa} {terr*1e6} BAT", file=myf)

def write_par(filename, F0, F1, PEPOCH, F1err=1e-13, F0err=1e-7, fit_omgdot=True):
    # ED: added os.makedirs to create directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    #File name should end in '.par'
    with open(filename, 'w') as myf:
        print(f"{'PSRJ':15}FAKE", file=myf)
        print(f"{'RAJ':15}0", file=myf)
        print(f"{'DECJ':15}0", file=myf)
        print(f"{'F0':15}{F0} 1  {F0err}", file=myf)
        if fit_omgdot:
            fit=1
        else:
            fit=0
        if F1 is not None:
            print(f"{'F1':15}{F1:.10e} {fit} {F1err}", file=myf)
        print(f"{'PEPOCH':15}{PEPOCH}", file=myf)
        print("TRACK -2", file=myf)