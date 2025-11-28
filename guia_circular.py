"""
Autor: Sebastián Aguilar

Cálculo y visualización de modos TE y TM en una guía de onda de sección circular.
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def TE_modes(m,n, R, rho, phi, z):
    """Calcula los campos de los modos Transversales Eléctricos (TE_mn) 
    en una guía de onda cilíndrica de radio R.
    Esta función utiliza coordenadas cilíndricas (rho, phi, z) y asume que 
    el medio es el vacío o aire (c = velocidad de la luz en el vacío).

    Args:
            m (int): Índice azimutal del modo (orden de la función de Bessel y su derivada).
            n (int): Índice radial del modo (n-ésimo cero de la derivada de la función de Bessel de orden m).
            R (float): Radio de la guía de onda cilíndrica.
            rho (float): Coordenada radial en el sistema cilíndrico.
            phi (float): Coordenada angular en el sistema cilíndrico.
            z (float): Coordenada axial en el sistema cilíndrico.

    Returns:
            tuple[np.ndarray]: Una tupla conteniendo los seis componentes de campo complejos:
                                                    (E_rho, E_phi, E_z, B_rho, B_phi, B_z).
                                                    E_z siempre es cero para los modos TE.
    Notas:
            * Los campos se normalizan por B0.
            * Las componentes de campo complejo incluyen la dependencia temporal
                y espacial $e^{j(kz - omega t)}$, donde j es la unidad imaginaria.
            * La frecuencia (omega) se establece automáticamente en 1.5 veces
                la frecuencia de corte del modo TE_mn.
    """
    # Constantes
    c = sp.constants.c # Velocidad de la luz en el vacío (m/s)
     
    # Funciónes de Bessel y ceros
    beta_mn = sp.special.jnp_zeros(m, n)[-1] # n-ésimo cero de la derivada de la función de Bessel de orden m
    J_m     = sp.special.jv(m, beta_mn * rho / R) # Función de Bessel de orden m
    J_mp    = sp.special.jvp(m, beta_mn * rho / R) # Derivada de la función de Bessel de orden m
     
    # Cálculo de la frecuencia de corte
    w_cutoff = (c / R) * beta_mn
    print(f"Frecuencia de corte TE{m}{n}: {w_cutoff/(2*np.pi*1e9):.2f} GHz")
     
    # Frecuencia y tiempo de operación
    omega = 1.5 * w_cutoff  # Frecuencia de operación
    k     = np.sqrt((omega / c)**2 - (beta_mn / R)**2) # Número de onda
    T     = 2 * np.pi / omega # Período de la onda
    t     = T/2 # Tiempo de evaluación
     
    # Cálculo de los campos normalizados por B0
    Erho_norm = - (1j * omega * m * R**2) / (beta_mn**2 * rho) * J_m * np.sin(m * phi) * np.exp(1j * k * z - 1j * omega * t) # E_rho/B_0
    Ephi_norm = - (1j * omega * R) / (beta_mn) * J_mp * np.cos(m * phi) * np.exp(1j * k * z - 1j * omega * t) # E_phi/B_0
    Ez_norm   = np.zeros_like(Erho_norm) # E_z/B_0
    Brho_norm = -(k/omega) * Ephi_norm # B_rho/B_0
    Bphi_norm = (k/omega) * Erho_norm # B_phi/B_0
    Bz_norm   = J_m * np.cos(m * phi) * np.exp(1j * k * z - 1j * omega * t) # B_z/B_0
    
    return Erho_norm, Ephi_norm, Ez_norm, Brho_norm, Bphi_norm, Bz_norm

def TM_modes(m,n, R, rho, phi, z):
    """Calcula los campos de los modos Transversales Magnéticos (TM_mn) 
    en una guía de onda cilíndrica de radio R.
    Esta función utiliza coordenadas cilíndricas (rho, phi, z) y asume que 
    el medio es el vacío o aire (c = velocidad de la luz en el vacío).

    Args:
            m (int): Índice azimutal del modo (orden de la función de Bessel y su derivada).
            n (int): Índice radial del modo (n-ésimo cero de la derivada de la función de Bessel de orden m).
            R (float): Radio de la guía de onda cilíndrica.
            rho (float): Coordenada radial en el sistema cilíndrico.
            phi (float): Coordenada angular en el sistema cilíndrico.
            z (float): Coordenada axial en el sistema cilíndrico.

    Returns:
            tuple[np.ndarray]: Una tupla conteniendo los seis componentes de campo complejos:
                                                    (E_rho, E_phi, E_z, B_rho, B_phi, B_z).
                                                    E_z siempre es cero para los modos TE.
    Notas:
            * Los campos se normalizan por E0.
            * Las componentes de campo complejo incluyen la dependencia temporal
                y espacial $e^{j(kz - omega t)}$, donde j es la unidad imaginaria.
            * La frecuencia (omega) se establece automáticamente en 1.5 veces
                la frecuencia de corte del modo TM_mn.
    """
    # Constantes
    c = sp.constants.c # Velocidad de la luz en el vacío (m/s)
     
    # Funciónes de Bessel y ceros
    alpha_mn = sp.special.jn_zeros(m, n)[-1] # n-ésimo cero de la función de Bessel de orden m
    J_m      = sp.special.jv(m, alpha_mn * rho / R) # Función de Bessel de orden m
    J_mp     = sp.special.jvp(m, alpha_mn * rho / R) # Derivada de la función de Bessel de orden m
     
    # Cálculo de la frecuencia de corte
    w_cutoff = (c / R) * alpha_mn
    print(f"Frecuencia de corte TM{m}{n}: {w_cutoff/(2*np.pi*1e9):.2f} GHz")
     
    # Frecuencia y tiempo de operación
    omega = 1.5 * w_cutoff  # Frecuencia de operación (rad/s)
    k     = np.sqrt((omega / c)**2 - (alpha_mn / R)**2) # Número de onda
    T     = 2 * np.pi / omega # Período de la onda
    t     = T/2 # Tiempo de evaluación
     
    # Cálculo de los campos normalizados por E0
    Erho_norm = (1j * R / alpha_mn) * k * J_mp * np.cos(m * phi) * np.exp(1j * k * z - 1j * omega * t) # E_rho/E_0
    Ephi_norm = - (1j * R**2) / (alpha_mn**2) * (m * k / rho) * J_m * np.sin(m * phi) * np.exp(1j * k * z - 1j * omega * t) # E_phi/E_0
    Ez_norm   = J_m * np.cos(m * phi) * np.exp(1j * k * z - 1j * omega * t) # E_z/E_0
    Brho_norm = -((omega) / (k * c**2)) * Ephi_norm # B_rho/E_0
    Bphi_norm = ((omega) / (k * c**2)) * Erho_norm # B_phi/E_0
    Bz_norm   = np.zeros_like(Brho_norm) # B_z/E_0

    return Erho_norm, Ephi_norm, Ez_norm, Brho_norm, Bphi_norm, Bz_norm

def plot_modes(mode, m, n, R, N, z, show=True, save=False, fname=None):
    # Grid
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    mask = rho <= R
    
    # Cálculo de los campos para el modo elegido (Se asume que TE_modes y TM_modes están definidos)
    if mode == "TE":
        Erho, Ephi, Ez, Brho, Bphi, Bz = TE_modes(m, n, R, rho, phi, z)
    elif mode == "TM":
        Erho, Ephi, Ez, Brho, Bphi, Bz = TM_modes(m, n, R, rho, phi, z)
    
    # Componentes del campo E en coordenadas Cartesianas
    Ex = Erho * np.cos(phi) - Ephi * np.sin(phi)
    Ey = Erho * np.sin(phi) + Ephi * np.cos(phi)
    # Componentes del campo B en coordenadas Cartesianas
    Bx = Brho * np.cos(phi) - Bphi * np.sin(phi)
    By = Brho * np.sin(phi) + Bphi * np.cos(phi)
    
    # Cálculo de la norma de los campos
    E_norm = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    B_norm = np.sqrt(np.abs(Bx)**2 + np.abs(By)**2 + np.abs(Bz)**2)
    
    # Componentes reales normalizadas (para el streamplot)
    # Se usa 1e-12 para evitar división por cero en las regiones de norma ~ 0
    E_norm_safe = E_norm + 1e-12
    B_norm_safe = B_norm + 1e-12
    
    Exr = np.real(Ex) / E_norm_safe
    Eyr = np.real(Ey) / E_norm_safe
    Bxr = np.real(Bx) / B_norm_safe
    Byr = np.real(By) / B_norm_safe
    
    # Aplicar máscara fuera de la guía
    E_norm[~mask] = np.nan
    B_norm[~mask] = np.nan
    Exr[~mask] = np.nan
    Eyr[~mask] = np.nan
    Bxr[~mask] = np.nan
    Byr[~mask] = np.nan
    
    # ================================
    # Plotting
    # ================================
    
    # Ajustar figsize para un layout de 1x2. Se ha reducido de 10x8 a 10x5 para compactar.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Parámetros para streamplot
    density = 1
    linewidth = 1
    arrowsize = 1
    s_color = "w"
    
    # ------------------
    # Plot campo E
    # ------------------
    # Se usa vmax = np.nanmax(E_norm) para la escala, tal como estaba.
    pcm = ax1.pcolormesh(X/R, Y/R, E_norm, shading='auto', vmin=np.nanmin(E_norm), 
                         vmax=np.nanmax(E_norm), cmap="jet")
    
    # Crea el divisor y el eje para el colorbar, asegurando que sea del tamaño del plot
    divider1 = make_axes_locatable(ax1)
    cax = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pcm, cax=cax, label = r"$||E||/B_0$") # Etiqueta correcta según la figura
    
    # Flujo de campo E
    ax1.streamplot(X/R, Y/R, Exr, Eyr, density=density, linewidth=linewidth, arrowsize=arrowsize, color=s_color)
    
    # Círculo de la guía de onda
    circle = plt.Circle((0,0), 1, color='k', fill=False, linewidth=1)
    ax1.add_patch(circle)
    
    # Ajustes de Ejes
    ax1.set_xlabel('x/R')
    ax1.set_ylabel('y/R')
    ax1.set_aspect('equal')
    
    # ------------------
    # Plot campo B
    # ------------------
    # Se usa vmax = np.nanmax(B_norm) para la escala, tal como estaba.
    pcm2 = ax2.pcolormesh(X/R, Y/R, B_norm, shading='auto', vmin=np.nanmin(B_norm), 
                          vmax=np.nanmax(B_norm), cmap="jet")
    
    # Crea el divisor y el eje para el colorbar, asegurando que sea del tamaño del plot
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    
    # Se corrige la etiqueta del colorbar según la figura proporcionada
    if mode == "TE":
         fig.colorbar(pcm2, cax=cax2, label = r"$||B||/B_0$")
    elif mode == "TM":
         fig.colorbar(pcm2, cax=cax2, label = r"$||B||/E_0$")
    
    # Flujo de campo B
    ax2.streamplot(X/R, Y/R, Bxr, Byr, density=density, linewidth=linewidth, arrowsize=arrowsize, color=s_color)
    
    # Círculo de la guía de onda
    circle = plt.Circle((0,0), 1, color='k', fill=False, linewidth=1)
    ax2.add_patch(circle)
    
    # Ajustes de Ejes
    ax2.set_xlabel('x/R')
    ax2.set_ylabel('y/R')
    ax2.set_aspect('equal')
    
    # Ajuste final
    plt.tight_layout()
    
    if save:
        if fname == None:
            fname = f"{mode}{m}{n}_fields.png"
        plt.savefig(fname, format="png", dpi=300)
        
    if show:
        plt.show()
        
if __name__ == "__main__":
	R = 0.2  # Radio de la guía circular (m)
	N = 500 # Número de puntos en cada dirección para el grid de evaluación
	
	# Ejemplo de uso: Graficar el modo TE01
	plot_modes(mode="TE", m=0, n=1, R=R, N=N, z=0, show=True, save=True)