import numpy as np
import math as mt
import matplotlib.pyplot as plt
import logging

"""
File name: MolecularDynamics.py
Author: Tijin Hanno Geo Saji
Student Number: 5010322
Date Created: 30/05/2021
Date last modified: 15/06/2021
"""

# reduced units
A = 1e-10
kb = 0.008314
sig = 3.73
eps = 148 * kb
NA = 6.022 * 1e23
R = kb * NA

logging.basicConfig(filename='output.dat', level=logging.DEBUG)


class Mol_Dynamics:
    def __init__(self, rho, box, temp, ndim, cutoff, dt, Q):
        self.mass = 16.04       # mass of methane molecule in g/mol
        self.box = box          # box size
        self.temp = temp        # temperature
        self.ndim = ndim        # dimension of the box
        self.cutoff = cutoff    # cutoff
        self.V = self.box*self.box*self.box     # volume of simulation box
        self.rho = rho          # density
        self.dt = dt            # time step
        self.Q = Q              

    def PBC(self, c):
        """
        Function to operate on the array passed the PBC and MIC conditions
        :param c: Array passed
        :return: PBC/MIC corrected array
        """
        nPart = len(c)
        for i in range(nPart):
            for j in range(i + 1, nPart):
                # compute distance and adjust for PBC
                xyz = c[i, :] - c[j, :]
                for k, f in enumerate(xyz):
                    if f < -self.box / 2:
                        n = mt.floor((-self.box / 2 - f) / self.box) + 1
                        xyz[k] += self.box * n
                    elif f > self.box / 2:
                        n = mt.floor((f - self.box / 2) / self.box) + 1
                        xyz[k] -= self.box * n
        return c

    def rho_to_Num(self):
        """
        Function to calculate number of particles from the density
        :return: Number of particles
        """
        N = float(self.rho) * (int(self.box) ** 3) * 3.75 * (10 ** (-5))
        # the calculation for the expression is as follows; units are shown inside square brackets
        # N_part = rho[kg m^-3] * (L_box [10^-10 m])^3 * 6.023*10^23[molecules/mol]/ 16.04 [ 10^-3 kg/mol]
        # N_part = rho*(L_box**3)*3.755*(10**(-5)) [molecules]
        # print("Number of methane particles is: ", int(N))
        return int(N)

    @staticmethod
    def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
        """
        radial pair distribution function

        :param xyz: coordinates in xyz format per frame
        :param LxLyLz: box length in vector format
        :param n_bins: number of bins
        :param r_range: range on which to compute rdf
        :return:
        """

        g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
        g_r[0] = 0
        g_r = g_r.astype(np.float64)
        rho = 0

        for i, xyz_i in enumerate(xyz):
            xyz_j = np.vstack([xyz[:i], xyz[i + 1:]])
            d = np.abs(xyz_i - xyz_j)
            d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
            d = np.sqrt(np.sum(d ** 2, axis=-1))
            temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
            g_r += temp_g_r

        rho += (i + 1) / np.prod(LxLyLz)
        r = 0.5 * (edges[1:] + edges[:-1])
        V = 4./3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
        norm = rho * i
        g_r /= norm * V

        return r, g_r

    def initGrid(self):
        """
        Function to initialize a 3D array for the system
        :return: Array with the coordinates of the system
        """
        nPart = self.rho_to_Num()
        # load empty array for coordinates
        coords = np.zeros((nPart, 3))
        # Find number of particles in each lattice line
        nstart, nstop = 1/16 * self.box, 15/16 * self.box
        n = int(nPart ** (1 / 3))
        # define lattice spacing
        spac = (nstop - nstart) / n
        # initiate lattice indexing
        index = np.zeros(3)
        # assign particle positions
        for part in range(nPart):
                coords[part, :] = (index * spac) + nstart
                # advance particle position
                index[0] += 1
                # if last lattice point is reached jump to next line
                if index[0] == n:
                    index[0] = 0
                    index[1] += 1
                    if index[1] == n:
                        index[1] = 0
                        index[2] += 1
        return coords - self.box/2, self.box

    def initVel(self, coords):
        """
        Function to initialize the initial velocities of the particles of the system and to set COM velocity to zero
        :param coords: Coordinates of the particles in the system
        :return: Initial velocity of the system
        """
        N_part = len(coords)
        vels = np.random.uniform(0, 1, size=(N_part, self.ndim))  # giving random velocities
        vels_magn2 = np.sum(np.square(vels), axis=1)
        sumv = np.sum(np.sqrt(vels_magn2), axis=0)          # sum of all velocities
        sumv2 = np.sum(vels_magn2, axis=0)                # sum of kinetic energy/mass
        sumv = sumv/N_part                  # to find velocity of centre of mass
        sumv2 = sumv2/N_part                 # mean square velocity
        fs = mt.sqrt(self.ndim * kb * self.temp / (self.mass * sumv2 * 1e4))   # scale factor
        vels = (vels-sumv) * fs           # shifting the centre of mass velocity to zero
        return vels

    def LJ_force(self, coords):
        """
        Function to calculate the inter-particle forces in the system using the LJ potential
        :param coords: Coordinates of the particles in the system
        :return: Array with the inter-particle forces in the system
        """
        # initialize empty array for forces
        forces = np.zeros(coords.shape)
        # obtain number of particles
        nPart = np.shape(coords)[0]
        cutoff2 = self.cutoff * self.cutoff
        sig6 = sig ** 6
        sig12 = sig6 * sig6
        lj1 = 48 * eps * sig12
        lj2 = 24 * eps * sig6

        # compute forces between all particles
        for i in range(nPart-1):
            for j in range(i + 1, nPart):
                # compute distance and adjust for PBC
                xyz = coords[i, :] - coords[j, :]
                for k, c in enumerate(xyz):
                    if c < -self.box / 2:
                        n = mt.floor((-self.box / 2 - c) / self.box) + 1
                        xyz[k] += self.box * n
                    elif c > self.box / 2:
                        n = mt.floor((c - self.box / 2) / self.box) + 1
                        xyz[k] -= self.box * n
                vec = xyz
                rij2 = (vec ** 2).sum(axis=-1)
                if rij2 < cutoff2:
                    r2inv = 1 / rij2
                    r6inv = r2inv * r2inv * r2inv
                    force = (r2inv * r6inv * (lj1 * r6inv - lj2))
                    forces[i, :] += vec * force
                    forces[j, :] -= vec * force
                else:
                    forces[i, :] += vec * 0
                    forces[j, :] -= vec * 0
        return forces

    def PE_Pressure(self, coords):
        """
        Function to calculate the Potential Energy and the Pressure of the system
        :param coords: Coordinates of the particles in the system
        :return: Potential Energy and Pressure
        """
        N = self.rho_to_Num()
        num_density = N / self.V
        Potential_energy = 0
        P_int = 0   # Interaction part of virial pressure
        nPart = coords.shape[0]
        cutoff2 = self.cutoff * self.cutoff
        # Try to keep the code DRY
        for i in range(nPart):
            for j in range(i + 1, nPart):
                xyz = coords[i, :] - coords[j, :]
                for k, c in enumerate(xyz):
                    if c < -self.box / 2:
                        n = mt.floor((-self.box / 2 - c) / self.box) + 1
                        xyz[k] += self.box * n
                    elif c > self.box / 2:
                        n = mt.floor((c - self.box / 2) / self.box) + 1
                        xyz[k] -= self.box * n
                vec = xyz
                rij2 = (vec ** 2).sum(axis=-1)
                sig6 = sig ** 6
                sig12 = sig6 * sig6
                if rij2 < cutoff2:
                    r2inv = 1 / rij2
                    r6inv = r2inv * r2inv * r2inv
                    r12inv = r6inv * r6inv
                    Potential_energy += 4 * eps * ((sig12 * r12inv) - (sig6 * r6inv))
                    P_int += 4 * eps * ((-12 * sig12 * r12inv) + (6 * sig6 * r6inv))   # Config. contribution of Pressure
        pressure = (num_density * kb * self.temp) - (P_int / (3 * self.V))  # final total pressure
        return Potential_energy, pressure

    def Temp_KE(self, vels):
        """
        Fuction to calculate the Kinetic energy and Temperature of the system
        :param vels: Velocities of the particles in the system
        :return: Kinetic Energy and Temperature
        """
        nPart = vels.shape[0]
        vels2 = np.sum(np.power(vels, 2))
        kinetic_Energy = (0.5 * self.mass * vels2) * 1e4
        Temperature = kinetic_Energy / (0.5 * self.ndim * nPart * kb)
        return kinetic_Energy, Temperature

    def velocity_Verlet(self, position, velocity, forces):
        """
        Function to implement the velocity-verlet integrator
        :param position: Coordinates of the particles in the system
        :param velocity: Velocities of the particles in the system
        :param forces: Inter-particle forces between the particles in the system
        :return: Position, Velocity and forces of the system for the next time step
        """
        dt2 = self.dt * self.dt
        imass = 1/self.mass
        position += self.dt * velocity + (dt2 * 0.5 * forces * imass * 1e-4)  # updating position for next time step
        position = self.PBC(position)
        v_half = velocity + (self.dt * forces * 0.5 * imass * 1e-4)
        # updating forces
        forces = self.LJ_force(position)
        velocity = v_half + (self.dt * forces * 0.5 * imass * 1e-4)
        return position, velocity, forces

    def velocity_verlet_thermostat(self, position, velocity, forces, psi):
        """
        Function the implement the velocity-verlet integrator with the thermostat
        :param position: Coordinates of the particles in the system
        :param velocity: Velocities of the particles in the system
        :param forces: Inter-particle forces between the particles in the system
        :param psi:
        :return: Position, Velocity, forces and psi of the system for the next time step
        """
        dt2 = self.dt * self.dt
        imass = 1/self.mass
        nPart = position.shape[0]
        # updating position for next time step
        position += velocity * self.dt + dt2 * 0.5 * ((forces * imass * 1e-4) - (psi * velocity))
        position = self.PBC(position)
        KE = self.Temp_KE(velocity)[0]
        self.temp = self.Temp_KE(velocity)[1]
        psi_half = psi + (self.dt * (0.5 / self.Q) * ((KE/nPart) - (1.5 * kb * self.temp)))
        v_half = velocity + (0.5 * self.dt * ((imass * forces * 1e-4) - (psi_half * velocity)))
        # updating forces
        forces = self.LJ_force(position)
        # calculate psi and velocities at t+dt and update forces
        KE_half = self.Temp_KE(v_half)[0]
        self.temp = self.Temp_KE(v_half)[1]
        psi = psi_half + (self.dt * 0.5 / self.Q) * ((KE_half/nPart) - (1.5 * kb * self.temp))
        velocity = (v_half + (0.5 * self.dt * forces * imass * 1e-4)) / (1 + (0.5 * self.dt * psi))
        return position, velocity, forces, psi


if __name__ == '__main__':
    md1 = Mol_Dynamics(358.4, 30, 400, 3, 14, 1, 100)   # Class object for higher density
    md2 = Mol_Dynamics(1.6, 182, 400, 3, 50, 1, 100)    # Class object for lower density
    Position, _ = md1.initGrid()
    velocity = md1.initVel(Position)
    fcs = np.zeros(np.shape(Position))
    psi = 0
    Nfreq = 100
    run = 10000
    logging.debug('Init_temp: {}'.format(md1.temp))
    logging.debug('Rho: {}'.format(md1.rho))
    logging.debug('Q: {}'.format(md1.Q))
    Arr_Temp = []
    Arr_Press = []
    Arr_PE = []
    Arr_KE = []
    Arr_TE = []
    for step in range(run):
        # Position, velocity, fcs = md1.velocity_Verlet(Position, velocity, fcs)
        Position, velocity, fcs, psi = md1.velocity_verlet_thermostat(Position, velocity, fcs, psi)
        md1.temp = md1.Temp_KE(velocity)[1]
        Arr_Temp.append(md1.temp)
        vels2sum = vels2 = np.sum(np.power(velocity, 2))
        Pressure = md1.PE_Pressure(Position)[1]
        Arr_Press.append(Pressure)
        Potential_Energy = md1.PE_Pressure(Position)[0]
        Arr_PE.append(Potential_Energy)
        Kinetic_Energy = md1.Temp_KE(velocity)[0]
        Arr_KE.append(Kinetic_Energy)
        Total_energy = Kinetic_Energy + Potential_Energy
        Arr_TE.append(Total_energy)
        if step % Nfreq == 0:
            logging.debug('step: {}'.format(step))
            logging.debug('Vels2: {}'.format(vels2sum))
            logging.debug('Temp: {}'.format(md1.temp))
            logging.debug('Pressure: {}'.format(Pressure))
            logging.debug('PE: {}'.format(Potential_Energy))
            logging.debug('KE: {}'.format(Kinetic_Energy))
            logging.debug('TE: {}'.format(Total_energy))
            logging.debug('\n')
    print("Complete")

    # fig, axs = plt.subplots(4)
    # fig.suptitle('State variables: Temp, Pressure, KE, PE')
    # axs[0].plot(Arr_Temp)
    # axs[1].plot(Arr_Pressure)
    # axs[2].plot(Arr_KE)
    # axs[3].plot(Arr_PE)
    # plt.show()

    # plt.plot(Total_energy)
    # plt.show()

    # arr = np.loadtxt('0.5_rho')
    # r, gr = rdf(arr, 30)
    # plt.plot(gr)
    # plt.show()
