# Chanelle Friend CSC-490
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# INITIAL SETUP---
G = 6.67430e-11     # gravitational constant

# INITIAL CONDITIONS for BODY C---

# (a) one that gets captured in the orbit-- position: -3.5e8 | velocity: 324.55
d_AC_captured = -5.2e8                   # distance between BODY A & BODY C
r_C_captured = np.array([d_AC_captured,0])        # Position of BODY C
v_C_captured = np.array([0,324.55])      # Velocity of BODY C

# (b) one that escapes the system-- position: -2.5e7 | velocity: 1.93e3
d_AC_escapes = -2.5e7                   # distance between BODY A & BODY C
r_C_escapes = np.array([d_AC_escapes,0])        # Position of BODY C
v_C_escapes = np.array([0,2.93e3])      # Velocity of BODY C

# (c) one that creates visibly chaotic dynamics-- position: -2.5e7 | velocity: 1.33e3
d_AC_chaotic = -2.5e7                           # distance between BODY A & BODY C
r_C_chaotic = np.array([d_AC_chaotic,0])        # Position of BODY C
v_C_chaotic = np.array([0,1.33e3])              # velocity of BODY C

# TIME PARAMETERS
t_max = 3600 * 24 * 30     # 30 days
#dt = 60 * 1                   # 5 minutes in seconds
dt = 60 * 1
steps = int(t_max / dt)
time = np.arange(0,t_max,dt)

def grav_force(m1,m2,r1,r2):
    distance = np.linalg.norm(r2 - r1)
    if distance == 0:
        return np.array([0.0,0.0])
    return (G * m1 * m2 * (r2 - r1)) / (distance ** 3)

def angular_momentum(m, r, v):
    return m * np.cross(r,v)

def get_KE(m, v):
    return .5 * m * (np.linalg.norm(v)**2)

def get_PE(m1, m2, r1,r2):
    global G
    distance = np.linalg.norm(r2 - r1)
    return -(G * m1 * m2) / distance

# simulation loop
def run_three_body_sim(title, d_AC, r_C, v_C, calculate_lyapunov=False):
    print(title)
    global G, steps, time
    mass_A = 1.0e24  # mass of BODY A [kg]
    mass_B = 1.0e24  # mass of BODY B [kg]
    mass_C = 1.0e22  # mass of BODY C [kg]

    d_AB = 1.0e8  # distance between BODY A & BODY B
    collision_distance = 6.85e6 # collision distance

    # INITIAL CONDITIONS for BODY A && BODY B
    r_A = np.array([0.0, 0])  # Position of BODY A (fixed at origin)
    r_B = np.array([d_AB, 0])  # Position of BODY B

    v_A = np.array([0.0, 0])  # Velocity of BODY A (fixed at rest)
    v_B = np.array([0, 1.0e3])  # Velocity of BODY B (in the positive y-direction)

    # Initialize arrays to store positions
    positions_BODYA = np.zeros((steps, 2))
    positions_BODYB = np.zeros((steps, 2))
    positions_BODYC = np.zeros((steps, 2))
    positions_COM = np.zeros((steps, 2))

    # arrays to store acceleration
    acceleration_BODYB = np.zeros(steps)
    acceleration_BODYC = np.zeros(steps)

    # arrays to store values
    value_ANGULAR = np.zeros(steps)
    value_TOTAL_ENERGY = np.zeros(steps)

    if calculate_lyapunov:
        difference_c = 1.0e-8
        d_AC_prime = d_AC + difference_c
        r_C_prime = r_C + np.array([difference_c, 0.0])
        v_C_prime = v_C.copy()
        positions_BODYC_prime = np.zeros((steps, 2))
        separation_distance = np.zeros(steps)
        r_C_init = r_C.copy()                           # initial r_C
        r_C_prime_init = r_C_prime.copy()

        # Run a second simulation for C'
        r_A_prime = r_A.copy()
        r_B_prime = r_B.copy()
        v_A_prime = v_A.copy()
        v_B_prime = v_B.copy()

    for i in range(steps):
        #print("d_AC", d_AC)

        # COMPUTE FORCE & ACCELERATION
        force_AB = grav_force(mass_A, mass_B,r_A,r_B)
        force_BC = grav_force(mass_B, mass_C, r_B, r_C)
        force_AC = grav_force(mass_A, mass_C,r_A,r_C)

        a_BODYA = (force_AB + force_AC) / mass_A
        a_BODYB = (-force_AB + force_BC) / mass_B
        a_BODYC = (-force_AC - force_BC) / mass_C

        if calculate_lyapunov:
            force_AB_prime = grav_force(mass_A, mass_B, r_A_prime, r_B_prime)
            force_BC_prime = grav_force(mass_B, mass_C, r_B_prime, r_C_prime)
            force_AC_prime = grav_force(mass_A, mass_C, r_A_prime, r_C_prime)

            a_BODYA_prime = (force_AB_prime + force_AC_prime) / mass_A
            a_BODYB_prime = (-force_AB_prime + force_BC_prime) / mass_B
            a_BODYC_prime = (-force_AC_prime - force_BC_prime) / mass_C

        # UPDATE POSITIONS
        r_A += v_A * dt + 0.5 * a_BODYA * dt ** 2
        r_B += v_B * dt + 0.5 * a_BODYB * dt ** 2
        r_C += v_C * dt + 0.5 * a_BODYC * dt ** 2

        if calculate_lyapunov:
            r_A_prime += v_A_prime * dt + 0.5 * a_BODYA_prime * dt ** 2
            r_B_prime += v_B_prime * dt + 0.5 * a_BODYB_prime * dt ** 2
            r_C_prime += v_C_prime * dt + 0.5 * a_BODYC_prime * dt ** 2

        # Compute NEW FORCE from updated positions
        force_AB_new = grav_force(mass_A, mass_B, r_A, r_B)
        force_BC_new = grav_force(mass_B, mass_C, r_B, r_C)
        force_AC_new = grav_force(mass_A, mass_C, r_A, r_C)

        a_BODYA_new = (force_AB_new + force_AC_new) / mass_A
        a_BODYB_new = (-force_AB_new + force_BC_new) / mass_B
        a_BODYC_new = (-force_AC_new - force_BC_new) / mass_C

        if calculate_lyapunov:
            force_AB_prime_new = grav_force(mass_A, mass_B, r_A_prime, r_B_prime)
            force_BC_prime_new = grav_force(mass_B, mass_C, r_B_prime, r_C_prime)
            force_AC_prime_new = grav_force(mass_A, mass_C, r_A_prime, r_C_prime)

            a_BODYA_prime_new = (force_AB_new + force_AC_new) / mass_A
            a_BODYB_prime_new = (-force_AB_new + force_BC_new) / mass_B
            a_BODYC_prime_new = (-force_AC_new - force_BC_new) / mass_C

        # Update velocities using AVERAGE ACCELERATION
        v_A += 0.5 * (a_BODYA + a_BODYA_new) * dt
        v_B += 0.5 * (a_BODYB + a_BODYB_new) * dt
        v_C += 0.5 * (a_BODYC + a_BODYC_new) * dt

        if calculate_lyapunov:
            v_A_prime += 0.5 * (a_BODYA_prime + a_BODYA_prime_new) * dt
            v_B_prime += 0.5 * (a_BODYB_prime + a_BODYB_prime_new) * dt
            v_C_prime += 0.5 * (a_BODYC_prime + a_BODYC_prime_new) * dt

        # CENTER OF MASS CALCULATION
        r_COM = (mass_A * r_A + mass_B * r_B + mass_C * r_C) / (mass_A + mass_B + mass_C)

        # ANGULAR MOMENTUM CALCULATION
        L_A = angular_momentum(mass_A, r_A, v_A)
        L_B = angular_momentum(mass_B, r_B, v_B)
        L_C = angular_momentum(mass_C, r_C, v_C)

        L_total = L_A + L_B + L_C
        # TOTAL ENERGY CALCULATION

        # --> kinetic energy calculations
        KE_A = get_KE(mass_A,v_A)
        KE_B = get_KE(mass_B,v_B)
        KE_C = get_KE(mass_C,v_C)

        # --> potential energy calculations
        PE_AB = get_PE(mass_A,mass_B,r_A,r_B)
        PE_BC = get_PE(mass_B,mass_C,r_B,r_C)
        PE_AC = get_PE(mass_A,mass_C,r_A,r_C)

        PE_A = PE_AB + PE_AC
        PE_B = PE_AB + PE_BC
        PE_C = PE_AC + PE_BC

        # --> total!
        #total_energy = (KE_A + PE_A) + (KE_B + PE_B) + (KE_C + PE_C)
        total_energy = KE_A + KE_B + KE_C + PE_AB + PE_BC + PE_AC

        # update position arrays
        positions_BODYA[i] = r_A
        positions_BODYB[i] = r_B
        positions_BODYC[i] = r_C
        positions_COM[i] = r_COM

        # update values arrays
        value_ANGULAR[i] = L_total
        value_TOTAL_ENERGY[i] = total_energy

        # update acceleration arrays
        acceleration_BODYB[i] = np.linalg.norm(a_BODYB)
        acceleration_BODYC[i] = np.linalg.norm(a_BODYC)

        # collision detection
        if np.linalg.norm(r_A - r_B) < collision_distance:
            print(f"Collision detected between BODY A and BODY B at time {time[i]:.2f}s")
        if np.linalg.norm(r_A - r_C) < collision_distance:
            print(f"Collision detected between BODY A and BODY C at time {time[i]:.2f}s")
        if np.linalg.norm(r_B - r_C) < collision_distance:
            print(f"Collision detected between BODY B and BODY C at time {time[i]:.2f}s")

        # update distance between c' and c for lyapunov
        if calculate_lyapunov:
            positions_BODYC_prime[i] = r_C_prime
            separation_distance[i] = np.linalg.norm(r_C - r_C_prime)

    # Plot the orbits
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns layout

    # Plot the trajectories
    axes[0, 0].plot(positions_COM[:, 0], positions_COM[:, 1], label="Center of Mass", color='yellow',linestyle='dashed')
    axes[0, 0].plot(positions_BODYA[:, 0], positions_BODYA[:, 1], label="BODY A", color='purple')
    axes[0, 0].plot(positions_BODYB[:, 0], positions_BODYB[:, 1], label="BODY B", color='blue')
    axes[0, 0].plot(positions_BODYC[:, 0], positions_BODYC[:, 1], label="BODY C", color='red')
    axes[0, 0].scatter([0], [0], color='purple', label="BODY A (initial position)")
    axes[0, 0].scatter([d_AB], [0], color='blue', label="BODY B (initial position)")
    axes[0, 0].scatter([d_AC], [0], color='red', label="BODY C (initial position)")
    axes[0, 0].set_title(title)
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].legend()
    axes[0, 0].axis("equal")

    # Plot Angular Momentum
    axes[0, 1].plot(value_ANGULAR, label="Angular Momentum", color='green')
    axes[0, 1].set_title("Angular Momentum of BODY C in " + title)
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("L (kg⋅m²/s)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # plotting Total Energy
    axes[1, 0].plot(value_TOTAL_ENERGY, label="Total Energy", color='orange')
    axes[1, 0].set_title("Total Energy in " + title)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("Energy (J)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # plotting acceleration
    def make_colored_line(x, y, c, cmap='plasma'):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(c.min(), c.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(c[:-1])  # remove last point to match segments
        lc.set_linewidth(2)
        return lc

    #  position data for B and C
    xB, yB = positions_BODYB[:, 0], positions_BODYB[:, 1]
    xC, yC = positions_BODYC[:, 0], positions_BODYC[:, 1]

    lcB = make_colored_line(xB, yB, acceleration_BODYB, cmap='Blues')
    lcC = make_colored_line(xC, yC, acceleration_BODYC, cmap='Reds')

    axes[1,1].add_collection(lcB)
    axes[1,1].add_collection(lcC)

    # Add colorbars
    cb1 = plt.colorbar(lcB, ax=axes[1,1], orientation='vertical', fraction=0.046, pad=0.04)
    cb1.set_label("Acceleration Magnitude (BODY B)")
    cb2 = plt.colorbar(lcC, ax=axes[1,1], orientation='vertical', fraction=0.046, pad=0.10)
    cb2.set_label("Acceleration Magnitude (BODY C)")

    # Labels and formatting
    axes[1,1].set_title("Trajectories of BODY B & C Colored by Acceleration")
    axes[1,1].set_xlabel("X Position (m)")
    axes[1,1].set_ylabel("Y Position (m)")
    axes[1,1].axis("equal")
    axes[1,1].grid(True)

    # plotting Lyapunov!
    if calculate_lyapunov:

        axes[2, 0].plot(separation_distance, label="distance(m)", color='purple')
        axes[2, 0].set_title("Lyapunov Exponent Calculation")
        axes[2, 0].set_xlabel("time (s)")
        axes[2, 0].set_ylabel("distance (m)")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        axes[2, 1].plot(positions_BODYC_prime[:, 0], positions_BODYC_prime[:, 1], label="BODY C'", color='pink')
        axes[2, 1].plot(positions_BODYC[:, 0], positions_BODYC[:, 1], label="BODY C", color='red')
        axes[2, 1].scatter([d_AC_prime], [0], color='pink', label="BODY C' (initial position)")
        axes[2, 1].scatter([d_AC], [0], color='red', label="BODY C (initial position)")
        axes[2, 1].set_title("Trajectories of C and C' of " + title)
        axes[2, 1].set_xlabel("x (m)")
        axes[2, 1].set_ylabel("y (m)")
        axes[2, 1].legend()
        axes[2, 1].axis("equal")

    else:
        fig.delaxes(axes[2, 0])
        fig.delaxes(axes[2, 1])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

run_three_body_sim("Three Body Problem-- Captured in the Orbit", d_AC_captured, r_C_captured, v_C_captured)
run_three_body_sim("Three Body Problem-- Escaped the Orbit",d_AC_escapes, r_C_escapes, v_C_escapes)
run_three_body_sim("Three Body Problem-- Chaotic", d_AC_chaotic, r_C_chaotic, v_C_chaotic, True)







