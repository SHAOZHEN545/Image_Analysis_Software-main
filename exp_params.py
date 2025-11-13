'''Default experimental parameters.'''

TOF_ms = 5.
PIXEL_SIZE_um=6.
MAG=2.
ATOMIC_MASS=23.

TRAP_x_Hz=100.
TRAP_y_Hz=10.
TRAP_z_Hz=300.

kBoltz_SI=1.38e-23
m_proton_SI=1.672e-27


def TempFromSigma_uK(sigma,axis="x"):
    m=m_proton_SI*ATOMIC_MASS
    if axis=="x":
        omega=2*3.14159*TRAP_x_Hz
    elif axis=="y":
        omega=2*3.14159*TRAP_y_Hz
    elif axis=="z":
        omega=2*3.14159*TRAP_z_Hz
    else:
        print("Bad axis choise in temperature calculation")
        omega=0.
        
    
    return 1e6*(m*omega**2*sigma**2)/(kBoltz_SI*(1+(omega*TOF_ms*1e-3)**2))

