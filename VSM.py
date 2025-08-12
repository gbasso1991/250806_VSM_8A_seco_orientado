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
#%% Funciones
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
#%% Levanto Archivos Seco orientado
data_horiz = np.loadtxt(os.path.join('data','8A_seco_orientado_horiz.txt'), skiprows=12)
H_horiz = data_horiz[:, 0]  # Gauss
m_horiz = data_horiz[:, 1]  # emu

data_para = np.loadtxt(os.path.join('data','8A_seco_orientado_para.txt'), skiprows=12)
H_para = data_para[:, 0]  # Gauss
m_para = data_para[:, 1]  # emu

data_verti = np.loadtxt(os.path.join('data','8A_seco_orientado_vert.txt'), skiprows=12)
H_verti = data_verti[:, 0]  # Gauss
m_verti = data_verti[:, 1]  # emu

#% PLOTEO ALL
fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_horiz, m_horiz, '.-', label='Horizontal')
a.plot(H_para, m_para, '.-', label='Paralelo')
a.plot(H_verti, m_verti, '.-', label='Vertical')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Muestra seca orientada')
plt.show()
# #% Descuento contribucion diamagnética del parafilm
# susceptibilidad_pfilm = -8.514e-7 # emu/g/G
# masa_pfilm = 0.00739 # g

# diamag_horiz= susceptibilidad_pfilm * masa_pfilm * H_horiz
# diamag_para = susceptibilidad_pfilm * masa_pfilm * H_para
# diamag_verti = susceptibilidad_pfilm * masa_pfilm * H_verti
# #% Descuento diamagnetismo
# m_horiz_new = m_horiz - diamag_horiz
# m_para_new = m_para - diamag_para
# m_verti_new = m_verti - diamag_verti
 
# fig, (a,b,c) = plt.subplots(3, 1, figsize=(8, 12), sharex=True, constrained_layout=True)

# # Horizontal
# a.plot(H_horiz, m_horiz, '.-', label='Original')
# a.plot(H_horiz, m_horiz_new, '.-', label='Corregido')
# a.plot(H_horiz, diamag_horiz, '.-', label='Parafilm')
# a.set_ylabel('m (emu)')
# a.set_title('Horizontal')
# a.legend()
# a.grid()

# # Paralelo
# b.plot(H_para, m_para, '.-', label='Original')
# b.plot(H_para, m_para_new, '.-', label='Corregido')
# b.plot(H_para, diamag_para, '.-', label='Parafilm')
# b.set_ylabel('m (emu)')
# b.set_title('Paralelo')
# b.legend()
# b.grid()

# Vertical
# c.plot(H_verti, m_verti, '.-', label='Original')
# c.plot(H_verti, m_verti_new, '.-', label='Corregido')
# c.plot(H_verti, diamag_verti, '.-', label='Parafilm')
# c.set_ylabel('m (emu)')
# c.set_title('Vertical')
# c.legend()
# c.grid()
# c.set_xlabel('H (G)')
# plt.show()

#% Normalizo por masa NPM
masa_NPM = 0.00028  # g
m_horiz_norm = m_horiz / masa_NPM
m_para_norm = m_para / masa_NPM 
m_verti_norm = m_verti / masa_NPM 

# Calculo de campo coercitivo
hc_horiz = coercive_field(H_horiz, m_horiz_norm)
hc_para = coercive_field(H_para, m_para_norm)
hc_verti = coercive_field(H_verti, m_verti_norm)

print(f'Horiz: {hc_horiz:.1uS} G')
print(f'Para: {hc_para:.1uS} G')
print(f'Verti: {hc_verti:.1uS} G')

#PLOTEO NORMALIZADO por masa NPM
fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_horiz, m_horiz_norm, '-', label=f'Horizontal\nHc={hc_horiz:.1uS} G')
a.plot(H_para, m_para_norm, '-', label=f'Paralelo\nHc={hc_para:.1uS}  G')
a.plot(H_verti, m_verti_norm, '-', label=f'Vertical\nHc={hc_verti:.1uS} G')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Secado c/ iman y orientado - Normalizado por masa NPM')
plt.savefig('8A_seco_orientado_h_p_v.png', dpi=300)
plt.show()
#% Aux 
fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_para, m_para_norm, '.-', label=f'Paralelo\nHc={hc_para:.1uS}  G')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Secado c/ iman y orientado')
plt.show()

# Fitting 
# Para almacenar resultados para graficar luego
ajustes_seco_orientado = []

for nombre, H, m in [
    ('horiz', H_horiz, m_horiz_norm), 
    ('para',  H_para,  m_para_norm),
    ('verti', H_verti, m_verti_norm)]:
    
    # Obtener curva anhistérica
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    
    # Crear sesión de ajuste
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=False)
    
    # Primer ajuste con mu y sigma fijos
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()

    # Segundo ajuste liberando mu y sigma
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')  # pesos por separación
    fit.fit()
    fit.update()

    # Guardar ajuste
    fit.save()
    
    # Mostrar parámetros ajustados y derivados
    fit.print_pars()
    
    # Mostrar parámetros derivados con unidad
    pars = fit.derived_parameters()
    for key, val in pars.items():
        unit = fit3.session.units.get(key, '')
    print(f"{key:15s} = {val} {unit}")    
    # Guardar los datos para graficar luego
    H_fit = fit.X
    m_ah = fit.Y
    m_fit = fit.Yfit  # resultado del ajuste
    m_fit_sin_diamag = m_fit - H_fit*fit.params['C'].value - fit.params['dc'].value
    ajustes_seco_orientado.append((nombre, H_fit, m_ah, m_fit, m_fit_sin_diamag))

plt.figure(figsize=(10, 6),constrained_layout=True)
for nombre, H, m_exp, m_fit in ajustes_seco_orientado:
    plt.plot(H, m_exp, 'o', label=f'{nombre} exp', alpha=0.5)
    plt.plot(H, m_fit, '-', label=f'{nombre} fit')

plt.xlabel('Campo magnético H (G)')
plt.ylabel('Magnetización M (emu/g)')
plt.legend()
plt.title('Comparación de ajuste vs datos experimentales')
plt.grid(True)
plt.show()

#%% 8A Seco
data_8A_seco = np.loadtxt(os.path.join('data','8A_seco_pfilm.txt'), skiprows=12)
H_8A_seco = data_8A_seco[:, 0]  # Gauss
m_8A_seco = data_8A_seco[:, 1]
masa_8A_seco=0.00028 #g

H_8A_seco = data_8A_seco[:, 0]  # Gauss
m_8A_seco = data_8A_seco[:, 1]  # emu
m_8A_seco_norm = m_8A_seco / masa_8A_seco  # Normalizo por masa
fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
#ax.plot(H_parafilm, m_parafilm, '.-', label='Parafilm')
ax.plot(H_8A_seco, m_8A_seco_norm, '.-', label='8A seco')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

H_seco_anhist, m_seco_anhist = mt.anhysteretic(H_8A_seco, m_8A_seco_norm)

ajustes_seco=[]
# Ajuste para seco
fit_seco = fit3.session(H_seco_anhist, m_seco_anhist, fname='seco', divbymass=False)

# Primer ajuste con mu y sigma fijos
fit_seco.fix('sig0')
fit_seco.fix('mu0')
fit_seco.free('dc')
fit_seco.fit()
fit_seco.update()

# Segundo ajuste liberando mu y sigma
fit_seco.free('sig0')
fit_seco.free('mu0')
fit_seco.set_yE_as('sep')  # pesos por separación
fit_seco.fit()
fit_seco.update()
fit_seco.save()
fit_seco.print_pars()
pars = fit_seco.derived_parameters()
for key, val in pars.items():
    unit = fit3.session.units.get(key, '')
    print(f"{key:15s} = {val} {unit}")

H_fit_seco = fit_seco.X
m_ah_seco = fit_seco.Y
m_fit_seco = fit_seco.Yfit  # resultado del ajuste
m_seco_fit_sin_diamag= m_fit_seco - H_fit_seco*fit_seco.params['C'].value -fit_seco.params['dc'].value # Resto de diamagnetismo

ajustes_seco.append(('seco', H_fit_seco, m_ah_seco, m_fit_seco,m_seco_fit_sin_diamag))

# Extraer valores para el cuadro de texto
ms = pars['m_s']
mu_mu = pars['<mu>_mu']
ms_str = f"$M_s$ = {ms:.2uP} emu/g"
mu_mu_str = f"$<\\mu>_\\mu$ = {mu_mu:.2uP}" if mu_mu is not None else ""
ajuste_text = ms_str + "\n" + mu_mu_str

plt.figure(figsize=(8,5), constrained_layout=True)

for nombre, H, m_exp, m_fit,m_fit_sd in ajustes_seco:
    plt.plot(H, m_exp, 'o', label=f'{nombre} exp', alpha=0.5)
    plt.plot(H, m_fit, '-', label=f'{nombre} fit')
    plt.plot(H, m_fit_sd, '-', label=f'fit sd')

plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.legend()
plt.title('Comparación de ajuste vs datos experimentales - Seco')
plt.grid(True)
# Agregar cuadro de texto a la derecha
plt.gca().text(
    0.75, 0.5, ajuste_text, transform=plt.gca().transAxes,
    fontsize=11, va='center', ha='center',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
plt.show()

#%% 8A FF 
data_8A_FF = np.loadtxt(os.path.join('data', '8A_FF.txt'), skiprows=12)
H_8A_FF = data_8A_FF[:, 0]  # Gauss
m_8A_FF = data_8A_FF[:, 1]  # emu
masa_8A_FF = 0.0496  # g (masa total de muestra)
C_mm = 10/1000  # concentración masa/mL (g/mL), 10 mg/mL = 0.01 g/mL

# Normalizo por masa de NPs en la muestra (m_emu / masa_NP)
# masa_NP = masa_total * C_mm (en g)
masa_NP_FF = masa_8A_FF * C_mm  # g de NPs en la muestra
m_8A_FF_norm = m_8A_FF / masa_NP_FF

fig2, ax2 = plt.subplots(figsize=(6,4), constrained_layout=True)
ax2.plot(H_8A_FF, m_8A_FF_norm, '.-', label='8A FF')
ax2.legend(ncol=1)
ax2.grid()
ax2.set_ylabel('m (emu/g NPs)')
plt.xlabel('H (G)')
plt.show()

H_FF_anhist, m_FF_anhist = mt.anhysteretic(H_8A_FF, m_8A_FF_norm)

ajustes_FF = []
fit_FF = fit3.session(H_FF_anhist, m_FF_anhist, fname='FF', divbymass=False)

# Primer ajuste con mu y sigma fijos
fit_FF.fix('sig0')
fit_FF.fix('mu0')
fit_FF.free('dc')
fit_FF.fit()
fit_FF.update()

# Segundo ajuste liberando mu y sigma
fit_FF.free('sig0')
fit_FF.free('mu0')
fit_FF.set_yE_as('sep')
fit_FF.fit()
fit_FF.update()
fit_FF.save()
fit_FF.print_pars()
pars_FF = fit_FF.derived_parameters()
for key, val in pars_FF.items():
    unit = fit3.session.units.get(key, '')
    print(f"{key:15s} = {val} {unit}")

H_fit_FF = fit_FF.X
m_ah_FF = fit_FF.Y
m_fit_FF = fit_FF.Yfit
m_FF_fit_sin_diamag = m_fit_FF - H_fit_FF*fit_FF.params['C'].value - fit_FF.params['dc'].value

ajustes_FF.append(('FF', H_fit_FF, m_ah_FF, m_fit_FF, m_FF_fit_sin_diamag))

# Extraer valores para el cuadro de texto
ms_FF = pars_FF['m_s']
mu_mu_FF = pars_FF['<mu>_mu']
ms_str_FF = f"$M_s$ = {ms_FF:.2uP} emu/g"
mu_mu_str_FF = f"$<\\mu>_\\mu$ = {mu_mu_FF:.2uP} \mu_B" if mu_mu_FF is not None else ""
ajuste_text_FF = ms_str_FF + "\n" + mu_mu_str_FF
#%%
plt.figure(figsize=(8,5), constrained_layout=True)
for nombre, H, m_exp, m_fit, m_fit_sd in ajustes_FF:
    #plt.plot(H, m_exp, 'o', label=f'{nombre} exp', alpha=0.5)
    plt.plot(H, m_fit, '-', label=f'{nombre} fit')
    plt.plot(H, m_fit_sd, '-', label=f'fit sd')
plt.plot(H_8A_FF,m_8A_FF_norm,'.-', label='8A FF exp', alpha=0.5)
plt.xlabel('H (G)')
plt.ylabel('m (emu/g NPs)')
plt.legend()
plt.title('Comparación de ajuste vs datos experimentales - FF')
plt.grid(True)
plt.gca().text(
    0.75, 0.5, ajuste_text_FF, transform=plt.gca().transAxes,
    fontsize=11, va='center', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
plt.show()

# %% 
#PLOTEO NORMALIZADO al maximo valor 
fig, a= plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)
a.plot(H_horiz, m_horiz_norm/max(m_horiz_norm), '-', label=f'Seco Horizontal')
a.plot(H_para, m_para_norm/max( m_para_norm), '-', label=f'Seco Paralelo')
a.plot(H_verti, m_verti_norm/max(m_verti_norm), '-', label=f'Seco Vertical')
a.set_xlabel('G')
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('8A - Secado c/ iman y orientado - normalizado a valor maximo')
#plt.savefig('8A_seco_orientado_h_p_v.png', dpi=300)
plt.show()
#%% COmparo 8A orientado h,p,v con FF
# Comparación de m normalizados descontando la contribución diamagnética

fig, a = plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, constrained_layout=True)

for nombre, H, m_exp, m_fit, m_fit_sd in ajustes_seco_orientado:
    a.plot(H, m_fit_sd / max(m_fit_sd), '-', label=f'Seco orientado {nombre} fit sd')

a.plot(H_8A_FF, m_8A_FF_norm / max(m_8A_FF_norm), '-', label='8A FF') 
a.plot(H_fit_seco, m_seco_fit_sin_diamag / max(m_seco_fit_sin_diamag), '-', label='8A seco sin orientar fit sd')
# # Seco orientado (horizontal, paralelo, vertical) - ya descontado diamagnetismo en m_fit_sin_diamag
# a.plot(H_horiz, (ajustes_seco_orientado[0][3] - H_horiz*fit3.session.units['C']`` - fit3.session.units['dc'])/max(ajustes_seco_orientado[0][3] - H_horiz*fit3.session.units['C'] - fit3.session.units['dc']), '-', c='C0', label='Horizontal fit sd')
# a.plot(H_para, (ajustes_seco_orientado[1][3] - H_para*fit3.session.units['C'] - fit3.session.units['dc'])/max(ajustes_seco_orientado[1][3] - H_para*fit3.session.units['C'] - fit3.session.units['dc']), '-', c='C1', label='Paralelo fit sd')
# a.plot(H_verti, (ajustes_seco_orientado[2][3] - H_verti*fit3.session.units['C'] - fit3.session.units['dc'])/max(ajustes_seco_orientado[2][3] - H_verti*fit3.session.units['C'] - fit3.session.units['dc']), '-', c='C2', label='Vertical fit sd')


# 8A FF y 8A seco sin orientar (ya tienes m_FF_fit_sin_diamag y m_seco_fit_sin_diamag)
#a.plot(H_fit_FF, m_FF_fit_sin_diamag / max(m_FF_fit_sin_diamag), '-', c='C3', label='8A FF fit sd')
#a.plot(H_fit_seco, m_seco_fit_sin_diamag / max(m_seco_fit_sin_diamag), '-', c='C4', label='8A seco sin orientar fit sd')
a.set_xlim(0,19e3)
a.set_ylim(0,1.1)
a.set_ylabel('m/m_s (sin diamagnetismo)')
a.set_xlabel('H (G)')
a.legend()
a.grid()
a.set_title('Comparación m normalizados (sin contribución diamagnética)')
plt.show()



