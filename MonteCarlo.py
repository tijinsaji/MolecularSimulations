import time
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import statistics as st
import copy

"""
File name: MonteCarlo.py
Author: Tijin Hanno Geo Saji
Student Number: 5010322
Date Created: 16/05/2021
Date last modified: 28/05/2021

"""


# defining constants
sig = 3.73 * 1e-10
eps = 204.24 * 1e-23  # from eps/kb = 148K
k = 1.38 * 1e-23


def mean(a):
    """
    Function to calculate the mean of the list/array passed
    :param a: List/array whose mean has to be passed
    :return: mean of the array passed
    """
    return sum(a) / len(a)


def PBC(d):
    """
    Function to calculate the Periodic boundary condition on the distance passed
    :param d: distance whose PBC has to be found
    :return: PBC corrected distance
    """
    return d % L_box_red


def PBC_MIC(d):
    """
    Function to calculate the Periodic boundary condition with Minimum image Convention on the distance passed
    :param d: distance whose PBC_MIC has to be found
    :return: PBC_MIC corrected distance
    """
    if d >= L_box_red*0.5:
        d -= L_box_red
    return d


def rho_to_Num():
    """
    Function to calculate the number of molecules within the simulation box from density and dimension of simulation box
    :return: Number of molecules
    """
    Num = float(rho) * (int(L_box) ** 3) * 3.75 * (10 ** (-5))
    # the calculation for the expression is as follows; units are shown inside square brackets
    # N_part = rho[kg m^-3] * (L_box [10^-10 m])^3 * 6.023*10^23[molecules/mol]/ 16.04 [ 10^-3 kg/mol]
    # N_part = rho*(L_box**3)*3.755*(10**(-5)) [molecules]
    return int(Num)


def startConf(N_part):
    """
    Function to initialize configuration for the system
    :param N_part: Number of molecules within the simulation box
    :return: Initial configuration of the system
    """
    coord = np.random.uniform(0.0, 1.0, size=(N_part, 3)) * L_box
    coord = coord.tolist()
    return coord


def total_Energy_Pressure(coord):
    """
    Function to return the total energy and pressure of the system whose coordinates are in the list 'coord'
    :param coord: List with the coordinates of molecules in the system
    :return: Total energy and pressure of the system
    """
    Energy = 0
    P_int = 0
    for i in range(len(coord) - 1):
        for j in range(i + 1, len(coord)):
            # finding the distance and applying the PBC/MIC conditions
            dx = PBC_MIC(abs(coord[i][0] - coord[j][0]))
            dy = PBC_MIC(abs(coord[i][1] - coord[j][1]))
            dz = PBC_MIC(abs(coord[i][2] - coord[j][2]))
            dist = mt.sqrt(dx * dx + dy * dy + dz * dz)  # distance between the two interacting particles
            dist2 = dist * dist
            dist6 = dist2 * dist2 * dist2
            dist6_inv = 1 / dist6
            dist12_inv = dist6_inv * dist6_inv
            if dist <= R_cut_red:
                Energy += 4 * (dist12_inv - dist6_inv)     # Potential energy of two particles interacting
                P_int += 4 * ((-12 * dist12_inv) + (6 * dist6_inv))   # Configuration contribution of Pressure
        u_tail = 8.37 * rho_red * (0.33 * R_cut_red9_inv - R_cut_red3_inv)  # energy tail corrections
        Energy += u_tail
        P_tail = 8.37 * rho_red * rho_red * (R_cut_red9_inv - R_cut_red3_inv)   # Pressure tail corrections
        P_int += P_tail
    Pressure = (rho_red * T_red) - (P_int / (3 * V_red))  # final total pressure
    return Energy, Pressure


def singleParticleEnergy(coord, particle_num):
    """
    Function to calculate the potential energy of the particle picked within the system
    :param coord: List with the coordinates of molecules in the system
    :param particle_num: Randomly chosen molecule from the system
    :return: Potential energy of the particle_num
    """
    Single_Energy = 0
    for j in range(len(coord)):
        if j != particle_num:
            # finding the distance between the particles and applying the PBC/MIC conditions
            dx = PBC_MIC(abs(coord[particle_num][0] - coord[j][0]))
            dy = PBC_MIC(abs(coord[particle_num][1] - coord[j][1]))
            dz = PBC_MIC(abs(coord[particle_num][2] - coord[j][2]))
            dist = mt.sqrt(dx * dx + dy * dy + dz * dz)
            dist2 = dist * dist
            dist6 = dist2 * dist2 * dist2
            dist6_inv = 1 / dist6
            dist12_inv = dist6_inv * dist6_inv
            if dist <= R_cut_red:
                Single_Energy += 4 * (dist12_inv - dist6_inv)   # Potential energy of two particles interacting
    u_tail = 8.37 * rho_red * (0.33 * R_cut_red9_inv - R_cut_red3_inv)   # energy tail corrections
    Single_Energy += u_tail
    return Single_Energy


def averages(a):
    """
    Function to calculate averages of the list/array passed
    :param a: List/array that is passed
    :return: average of a
    """
    Energy_average = mean(a)
    return Energy_average


def translate(coord, del_max, num_iter, Temp_red):
    """
    Function to perform translational trial moves
    :param coord: List of coordinates of molecules within the system
    :param del_max: Maximum allowed displacement of the coordinates
    :param num_iter: Number of iterations of the MC cycles to be done
    :param Temp_red: The temperature used for the translation
    :return: updated coord, number of accepted moves, total and average energy and pressure
    """
    accept = 0
    Energy_total = []
    Energy_average = []
    Pressure_average = []
    Pressure_total = []

    for i in range(num_iter):

        particle = np.random.randint(0, len(coord))  # will take values from 0 to N_part-1

        Uo = singleParticleEnergy(coord, particle)  # calculating the old potential energy of 'particle'

        coord_new = copy.deepcopy(coord)    # making a copy of the old coordinates in coord_new

        # translating x,y,z coordinates of coord_new by a random amount which depends on del_max
        coord_new[particle][0] = PBC(coord_new[particle][0] + (2*np.random.random() - 1) * del_max)
        coord_new[particle][1] = PBC(coord_new[particle][1] + (2*np.random.random() - 1) * del_max)
        coord_new[particle][2] = PBC(coord_new[particle][2] + (2*np.random.random() - 1) * del_max)

        Un = singleParticleEnergy(coord_new, particle)  # calculating the new potential energy of 'particle'

        beta = 1 / Temp_red
        del_U = Un - Uo
        del_U_beta = del_U * beta
        if del_U_beta < 75:   # to reject new configurations that have very high energy compared to old configuration
            if del_U_beta <= 0.0:       # if conditions are satisfied, copy new coordinates to old and add 1 to accept
                coord = copy.deepcopy(coord_new)
                accept += 1
            elif mt.exp(-del_U_beta) > np.random.random():  # comparing exp(-del_U_beta) with random number [0,1)
                coord = copy.deepcopy(coord_new)
                accept += 1
        if i % 10 == 0:         # sample averages every 10 cycles
            Energy_total.append(total_Energy_Pressure(coord)[0])
            Energy_average.append(averages(Energy_total))
            Pressure_total.append(total_Energy_Pressure(coord)[1])
            Energy_average.append(averages(Energy_total))
        if i !=0 and i % 10000 == 0:      # Print averages every 10000 cycles
            print("From iterations, Energy_avg", i, st.mean(Energy_average))
            print("From iterations, Energy_avg", i, st.mean(Pressure_average))
    return coord, accept, Energy_total, Energy_average, Pressure_total, Pressure_average


def initialization(Num_mol, del_max, n_iter, Temp):
    """
    Function to equilibrate the initial configuration
    :param Num_mol: Number of molecules in the system
    :param del_max: Maximum displacement allowed
    :param n_iter: Number of iteration to do to equilibrate the system
    :param Temp: Temperature used in the simulation
    :return: Coordinates of molecules in system after equilibration
    """
    coord = startConf(Num_mol)    # start an initial configuration
    coord = translate(coord, del_max, n_iter, Temp)[0]  # equilibrate the system through n_iter translational moves
    return coord


def optimum_disp(disp_i, Temp):
    """
    Function to calculate answers for Qn 2.1 and 2.2 in the assignment
    :param disp_i: Maximum displacement allowed
    :param Temp: Temperature used in the simulation
    :return: number of accepted moves
    """
    Coords = initialization(N, 0.5*1e-10/sig, N*50, Temp)   # Initialization is done with the max. allowed disp
    print("Initialization is over")
    acceptance = translate(Coords, disp_i, 500000, Temp)[1]
    print("Final results", acceptance, disp_i)
    print('\n')
    return acceptance


def energy_cycles(disp_i, Temp):
    """
    Function to calculate answers for Qn 2.3 in the assignment
    :param disp_i: Maximum displacement allowed
    :param Temp: Temperature used in the simulation
    :return: List with total energy and Avg. energy sampled after every 10 cycles
    """
    Coords = initialization(N, 0.5*1e-10/sig, N*50, Temp)       # Initialization is done with the max. allowed disp
    print("Initialization is over")
    _, _, Energy_total, Energy_average, _, _ = translate(Coords, disp_i, 500000, Temp)
    return Energy_total, Energy_average


def isotherm(disp_i, Temp):
    """
    Function to calculate answers for Qn 3.1 in the assignment
    :param disp_i: Maximum displacement allowed
    :param Temp: Temperature used in the simulation
    :return: List with total pressure and Avg. pressure sampled after every 10 cycles
    """
    Coords = initialization(N, 0.5*1e-10/sig, N*50, Temp)
    print("Initialization is over")
    _, _ , _, _, pressure, pressure_avg = translate(Coords, disp_i, 500000, Temp)
    print("Final results", pressure_avg, Temp)
    print('\n')
    return pressure, pressure_avg


if __name__ == '__main__':
    start = time.perf_counter()
    # **for Qn 2 and 3**
    # L_box, rho, T, R_cut = 30, 358.4, 150, 14                   # for Qn 2.1, 2.3
    # L_box, rho, T,  R_cut = 75, 9.68, 400, 30                 # for Qn 2.2
    # L_box, rho, R_cut = 30, 358.4, 14                         # for Qn 3.1 a
    # L_box, mass_density, R_cut = 182*1e-10, 1.6, 50*1e-10     # for Qn 3.1 b

    # N = rho_to_Num()
    # sig3 = sig*sig*sig
    # L_box_red = L_box/sig
    # R_cut_red = R_cut/sig
    # R_cut_red3 = R_cut_red * R_cut_red * R_cut_red
    # R_cut_red3_inv = 1 / R_cut_red3
    # R_cut_red9_inv = R_cut_red3_inv * R_cut_red3_inv * R_cut_red3_inv
    # V = L_box * L_box * L_box
    # V_red = L_box_red * L_box_red * L_box_red
    # rho_red = (N * sig3) / V
    # T_red = 6.756*1e-3 * T  # has units of [1/K], 6.756*1e-3 = 1/148 = eps/kb

    # ** for Qn 2**
    # dpl = [0.001, 0.05, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1]  # for Qn 2.1
    # dpl = [0.05, 0.1, 0.5, 1, 5, 10, 20, 25, 30, 35, 37]   # for Qn 2.2
    # acceptance = [optimum_disp(disp, T_red) for disp in dpl]

    # for qn 2.3
    # disp = 0.45 * 1e-10 / sig
    # E_average, E_total = energy_cycles(disp, T)
    # print("The average is :", mean(E_average))
    # # error is considered as the standard deviation of the average energy
    # error = st.stdev(E_average)
    # print(" The error is :", error)
    # plt.plot(E_total, label = "E_total")
    # plt.plot(E_average, label = "E_average")
    # plt.legend()
    # plt.show()
    # Et_array = np.array(E_total)
    # Ea_array = np.array(E_average)

    # for question 3.1 and 3.2
    # disp = 0.45 * 1e-10/sig  # for Qn 3.1
    # disp = 20 * 1e-10/sig  # for Qn 3.2
    # T_red = 200 * 6.756*1e-3
    # T_red = 300 * 6.756*1e-3
    # T_red = 400 * 6.756*1e-3
    # pressure_average = mean(isotherm(disp, T_red)[1])
    stop = time.perf_counter()
    timetaken = stop - start
    print("time taken is", timetaken)
