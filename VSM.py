#%% VSM muestra 8A de Pablo Tancredi - Agosto 2025
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
#%%
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve el valor medio del campo coercitivo (Hc) como ufloat, 
    imprime ambos valores de Hc encontrados.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_ufloat: ufloat con el valor medio y la diferencia absoluta como incertidumbre
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    if len(hc_values) != 2:
        print("Advertencia: Se encontraron {} valores de Hc, se esperaban 2.".format(len(hc_values)))
        return None

    print(f"Hc encontrados: {hc_values[0]:.3f}, {hc_values[1]:.3f}")

    # Valor medio considerando el signo
    hc_mean =abs((hc_values[0] - hc_values[1]) / 2)
    # Incertidumbre: diferencia entre los valores absolutos
    hc_unc = abs(abs(hc_values[0]) - abs(hc_values[1]))
    return ufloat(hc_mean, hc_unc)
#%% Levanto Archivos
data_horiz = np.loadtxt(os.path.join('data','8A_seco_orientado_horiz.txt'), skiprows=12)
H_horiz = data_horiz[:, 0]  # Gauss
m_horiz = data_horiz[:, 1]  # emu

data_para = np.loadtxt(os.path.join('data','8A_seco_orientado_para.txt'), skiprows=12)
H_para = data_para[:, 0]  # Gauss
m_para = data_para[:, 1]  # emu

data_verti = np.loadtxt(os.path.join('data','8A_seco_orientado_vert.txt'), skiprows=12)
H_verti = data_verti[:, 0]  # Gauss
m_verti = data_verti[:, 1]  # emu


#%% PLOTEO ALL
fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_horiz, m_horiz, '.-', label='Horizontal')
a.plot(H_para, m_para, '.-', label='Paralelo')
a.plot(H_verti, m_verti, '.-', label='Vertical')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Muestra seca orientada')

plt.show()
#%% Descuento contribucion diamagnética del parafilm
susceptibilidad_pfilm = -8.514e-7 # emu/g/G
masa_pfilm = 0.00739 # g

diamag_horiz= susceptibilidad_pfilm * masa_pfilm * H_horiz
diamag_para = susceptibilidad_pfilm * masa_pfilm * H_para
diamag_verti = susceptibilidad_pfilm * masa_pfilm * H_verti
#% Descuento diamagnetismo
m_horiz_new = m_horiz - diamag_horiz
m_para_new = m_para - diamag_para
m_verti_new = m_verti - diamag_verti
 
fig, (a,b,c) = plt.subplots(3, 1, figsize=(8, 12), sharex=True, constrained_layout=True)

# Horizontal
a.plot(H_horiz, m_horiz, '.-', label='Original')
a.plot(H_horiz, m_horiz_new, '.-', label='Corregido')
a.plot(H_horiz, diamag_horiz, '.-', label='Parafilm')
a.set_ylabel('m (emu)')
a.set_title('Horizontal')
a.legend()
a.grid()

# Paralelo
b.plot(H_para, m_para, '.-', label='Original')
b.plot(H_para, m_para_new, '.-', label='Corregido')
b.plot(H_para, diamag_para, '.-', label='Parafilm')
b.set_ylabel('m (emu)')
b.set_title('Paralelo')
b.legend()
b.grid()

# Vertical
c.plot(H_verti, m_verti, '.-', label='Original')
c.plot(H_verti, m_verti_new, '.-', label='Corregido')
c.plot(H_verti, diamag_verti, '.-', label='Parafilm')
c.set_ylabel('m (emu)')
c.set_title('Vertical')
c.legend()
c.grid()

axs[2].set_xlabel('H (G)')
plt.show()

#%% Normalizo por masa NPM
masa_NPM = 0.00028  # g
m_horiz_norm = m_horiz_new / masa_NPM
m_para_norm = m_para_new / masa_NPM 
m_verti_norm = m_verti_new / masa_NPM 

# Calculo de campo coercitivo
hc_horiz = coercive_field(H_horiz, m_horiz_norm)
hc_para = coercive_field(H_para, m_para_norm)
hc_verti = coercive_field(H_verti, m_verti_norm)

print(f'Horiz: {hc_horiz:.1uS} G')
print(f'Para: {hc_para:.1uS} G')
print(f'Verti: {hc_verti:.1uS} G')

#PLOTEO NORMALIZADO
fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_horiz, m_horiz_norm, '-', label=f'Horizontal\nHc={hc_horiz:.1uS} G')
a.plot(H_para, m_para_norm, '-', label=f'Paralelo\nHc={hc_para:.1uS}  G')
a.plot(H_verti, m_verti_norm, '-', label=f'Vertical\nHc={hc_verti:.1uS} G')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Secado c/ iman y orientado')
plt.savefig('8A_seco_orientado_h_p_v.png', dpi=300)
plt.show()
#%%


fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_para, m_para_norm, '.-', label=f'Paralelo\nHc={hc_para:.1uS}  G')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Secado c/ iman y orientado')
plt.show()

# %% Fiteo 
resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre, H, m in [
    ('horiz', H_horiz, m_horiz_norm), 
    ('verti', H_verti, m_verti_norm)]:
    
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=False)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Y - linear_contrib
    m_saturacion = ufloat(np.mean([max(m_fit_sin_lineal),-min(m_fit_sin_lineal)]), np.std([max(m_fit_sin_lineal),-min(m_fit_sin_lineal)]))
    resultados_fit[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Y,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':m_saturacion,
                            'fit': fit}
    
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = m_fit_sin_lineal

# %%
