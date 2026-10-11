# Theoretical Documentation — 2D Continuum SIMP & Buckling Topology Optimization Engine

Questo documento raccoglie in modo rigoroso tutte le fondamenta teoriche, matematiche e numeriche implementate nel motore di Ottimizzazione Topologica 2D (`physics_engine/simp_engine_2d.py`), nel solutore reticolare Ground Structure (`physics_engine/ground_structure.py`) e nella dashboard interattiva (`app.py`).

---

## 1. Invarianza della Topologia al Ribaltamento del Carico in Elasticità Lineare

Applicando una forza $\mathbf{F}$ o la sua esatta opposta $-\mathbf{F}$ in un'ottimizzazione puramente elastica ($\alpha = 0$), la topologia ottimizzata risulta identica. Ciò è una conseguenza diretta della teoria dell'**Elasticità Lineare**:

$$
\mathbf{K}(\hat{\mathbf{p}}) \mathbf{u} = \mathbf{F}
$$

Invertendo il carico in $-\mathbf{F}$, il campo di spostamenti diventa $-\mathbf{u}$ e la **Compliance** (energia di deformazione / lavoro esterno) rimane invariata:

$$
C_{\text{new}} = (-\mathbf{F})^T (-\mathbf{u}) = \mathbf{F}^T \mathbf{u} = C
$$

Poiché il materiale base è isotropo e simmetrico a trazione/compressione ($E_t = E_c = E_0$), la sola minimizzazione della compliance non distingue tra puntoni compressi (soggetti a instabilità) e tiranti tesi. Per rompere tale simmetria fisica e rinforzare selettivamente le membrature compresse, il motore integra l'**Analisi di Buckling Linearizzato (LBA)**.

---

## 2. Metodo agli Elementi Finiti (FEM) nel Continuo 2D

### 2.1 Elemento Finito Q4 in Stato Piano di Tensione (Plane Stress)
Il dominio di progetto rettangolare $\Omega \subset \mathbb{R}^2$ è discretizzato con una griglia strutturata di $m = \text{nelx} \times \text{nely}$ elementi quadrilateri bilineari a 4 nodi (**Q4**), ciascuno di dimensioni $\Delta x \times \Delta y$ e spessore fuori piano $t$:
- **Gradi di Libertà (GDL)**: 2 traslazioni $(u_x, u_y)$ per nodo $\implies 8$ GDL per elemento, $n = 2(\text{nelx}+1)(\text{nely}+1)$ GDL globali.
- **Coordinate naturali e Jacobiano**: Con riferimento centrato nel baricentro dell'elemento $(\xi, \eta) \in [-1, 1]^2$:
  $$
  \mathbf{J} = \begin{bmatrix} \Delta x / 2 & 0 \\ 0 & \Delta y / 2 \end{bmatrix}, \quad \det(\mathbf{J}) = \frac{\Delta x \, \Delta y}{4}
  $$
- **Funzioni di forma bilineari**:
  $$
  N_1 = \frac{1}{4}(1-\xi)(1-\eta), \quad N_2 = \frac{1}{4}(1+\xi)(1-\eta), \quad N_3 = \frac{1}{4}(1+\xi)(1+\eta), \quad N_4 = \frac{1}{4}(1-\xi)(1+\eta)
  $$

### 2.2 Matrice di Rigidezza Elementare Costante $\mathbf{k}_0$
La matrice costitutiva elastica in stato piano di tensione (con $E_0 = 1$) è:

$$
\mathbf{C}_0 = \frac{1}{1 - \nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}
$$

La matrice di rigidezza elementare solida $\mathbf{k}_0 \in \mathbb{R}^{8 \times 8}$ viene pre-calcolata una sola volta all'inizializzazione tramite **Quadratura di Gauss esatta $2 \times 2$**:

$$
\mathbf{k}_0 = t \int_{-1}^{1} \int_{-1}^{1} \mathbf{B}(\xi, \eta)^T \mathbf{C}_0 \mathbf{B}(\xi, \eta) \det(\mathbf{J}) \, d\xi \, d\eta
$$

---

## 3. Formulazione SIMP a Tre Campi: Filtro di Densità e Proiezione di Heaviside

Per evitare instabilità numeriche (*checkerboarding*), dipendenza dalla mesh e bordi grigi diffusi, il solutore implementa una rigorosa **formulazione a tre campi** ($\mathbf{x} \to \tilde{\mathbf{x}} \to \hat{\mathbf{p}}$):

1. **Variabili di Progetto Matematiche ($\mathbf{x} \in [0, 1]^m$)**: Aggiornate dall'ottimizzatore.
2. **Densità Filtrate ($\tilde{\mathbf{x}}$ — Filtro Conico Spaziale)**:
   $$
   \tilde{x}_e = \frac{\sum_{i \in \mathcal{N}_e} H_{ei} x_i}{\sum_{i \in \mathcal{N}_e} H_{ei}} = \frac{(\mathbf{H}\mathbf{x})_e}{H_{s,e}}, \quad H_{ei} = \max\left(0, \, r_{\min} - \text{dist}(e, i)\right)
   $$
3. **Densità Fisiche Proiettate ($\hat{\mathbf{p}} \equiv \mathbf{x}_{\text{Phys}}$ — Proiezione Regolarizzata di Heaviside)**:
   $$
   \hat{p}_e(\tilde{x}_e) = \frac{\tanh(\beta \eta) + \tanh\left(\beta(\tilde{x}_e - \eta)\right)}{\tanh(\beta \eta) + \tanh\left(\beta(1 - \eta)\right)}
   $$
   con soglia $\eta = 0.5$ e schema di continuazione sul parametro di nitidezza $\beta = \min\left(16, \, 2^{\lfloor \text{it}/15 \rfloor}\right)$, che raddoppia ogni 15 iterazioni forzando la convergenza verso una topologia netta $0/1$.

### 3.1 Interpolazione Materiale SIMP Differenziata (Rigidezza vs. Stress Geometrico)
Per sopprimere i **modi di buckling spuri** nelle regioni a bassa densità ($\hat{p}_e \to 0$) secondo Ferrari & Sigmund (2020, 2021), la rigidezza elastica $\mathbf{K}$ utilizza un limite inferiore $E_{\min} = 10^{-9}$, mentre la rigidezza geometrica $\mathbf{G}$ è interpolata senza offset inferiore con esponente $p_G = 3.0$:

$$
E_K(\hat{p}_e) = E_{\min} + (E_0 - E_{\min}) \hat{p}_e^{p}, \qquad E_G(\hat{p}_e) = E_0 \hat{p}_e^{p_G}
$$

---

## 4. Analisi di Buckling Linearizzato (LBA) e Matrice Geometrica Green-Lagrange

### 4.1 Matrice di Rigidezza Geometrica $\mathbf{G}(\hat{\mathbf{p}}, \mathbf{u})$
Dalla linearizzazione del Principio dei Lavori Virtuali con il tensore di deformazione non lineare di **Green-Lagrange**, il lavoro degli sforzi pre-buckling dipende dai gradienti completi di *entrambe* le componenti di spostamento $(u, v)$:

$$
\varepsilon_{xx}^{NL} = \frac{1}{2}(u_{,x}^2 + v_{,x}^2), \quad \varepsilon_{yy}^{NL} = \frac{1}{2}(u_{,y}^2 + v_{,y}^2), \quad \gamma_{xy}^{NL} = u_{,x} u_{,y} + v_{,x} v_{,y}
$$

Definendo il vettore dei gradienti di spostamento $\mathbf{g} = [u_{,x}, v_{,x}, u_{,y}, v_{,y}]^T = \mathbf{B}_1 \mathbf{u}_e$, la matrice di rigidezza geometrica elementare valutata nel baricentro ($\xi = \eta = 0$) si decompone esattamente in tre matrici costanti pre-calcolate ($\mathbf{T}_x, \mathbf{T}_y, \mathbf{T}_{xy} \in \mathbb{R}^{8 \times 8}$):

$$
\mathbf{G}_e^0(\boldsymbol{\sigma}_e) = \sigma_{x,e} \mathbf{T}_x + \sigma_{y,e} \mathbf{T}_y + \tau_{xy,e} \mathbf{T}_{xy}
$$

dove:
$$
\mathbf{T}_x = \Omega_e \mathbf{B}_1^T \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix} \mathbf{B}_1, \quad
\mathbf{T}_y = \Omega_e \mathbf{B}_1^T \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \mathbf{B}_1, \quad
\mathbf{T}_{xy} = \Omega_e \mathbf{B}_1^T \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \mathbf{B}_1
$$

L'assemblaggio globale di $\mathbf{G} = \sum_e E_G(\hat{p}_e) \mathbf{G}_e^0(\boldsymbol{\sigma}_e)$ avviene in forma totalmente vettorializzata.

### 4.2 Formulazione agli Autovalori Reciproci ($\mu_i = 1/\lambda_i$)
Il problema agli autovalori generalizzato dell'instabilità linearizzata:

$$
\left(\mathbf{K}[\hat{\mathbf{p}}] + \lambda_i \mathbf{G}[\hat{\mathbf{p}}, \mathbf{u}]\right) \boldsymbol{\phi}_i = \mathbf{0}
$$

viene riformulato rispetto al reciproco del Buckling Load Factor $\mu_i = 1 / \lambda_i$:

$$
\mathbf{G}[\hat{\mathbf{p}}, \mathbf{u}] \boldsymbol{\phi}_i = -\mu_i \mathbf{K}[\hat{\mathbf{p}}] \boldsymbol{\phi}_i
$$

Massimizzare il carico critico fondamentale $\lambda_1 = \min_i \lambda_i > 0$ equivale quindi a **minimizzare il massimo autovalore reciproco** $\mu_{\max} = \max_i \mu_i$, con autovettori $\mathbf{K}$-ortonormalizzati ($\boldsymbol{\phi}_i^T \mathbf{K} \boldsymbol{\phi}_i = 1$).

---

## 5. Aggregazione Multimodale di Kreisselmeier-Steinhauser (KS) e Metodo Aggiunto

### 5.1 Prevenzione del Mode-Switching tramite Funzione KS
Durante l'ottimizzazione strutturale, gli autovalori di buckling $\mu_1, \mu_2, \dots, \mu_{n_{\text{eig}}}$ tendono a incrociarsi (*mode switching* / *eigenvalue coalescing*), rendendo $\max_i \mu_i$ non differenziabile e causando oscillazioni nella convergenza.

Il motore permette di scegliere tra:
1. **Modo Fondamentale Singolo ($n_{\text{eig}} = 1$)**: $\mu_{KS} = \mu_1$.
2. **Aggregatore KS Multimodale ($n_{\text{eig}} > 1$, default $n_{\text{eig}} = 4$)**: Estrae $n_{\text{eig}} + 4$ autovalori con Lanczos (`scipy.sparse.linalg.eigsh`) per robustezza sui cluster e aggrega i primi $n_{\text{eig}}$ modi tramite la funzione liscia di **Kreisselmeier-Steinhauser** traslata per stabilità numerica (evitando overflow esponenziali):

$$
\mu_{KS}(\boldsymbol{\mu}) = \mu_{\max} + \frac{1}{\rho_{KS}} \ln \left( \sum_{i=1}^{n_{\text{eig}}} \exp\left(\rho_{KS} (\mu_i - \mu_{\max})\right) \right), \qquad \mu_{\max} = \max_{1 \le i \le n_{\text{eig}}} \mu_i
$$

con parametro di aggregazione $\rho_{KS} = 40$.

### 5.2 Analisi di Sensitività tramite Stato Aggiunto e Fattorizzazione LU Riusabile
La derivata analitica del singolo autovalore $\mu_i$ rispetto alla densità fisica $\hat{p}_e$ richiede il vettore aggiunto $\mathbf{w}_i$ (per tenere conto della dipendenza implicita $\mathbf{G}(\mathbf{u}(\hat{\mathbf{p}}))$):

$$
\frac{\partial \mu_i}{\partial \hat{p}_e} = -\left[ \boldsymbol{\phi}_{i,e}^T \frac{\partial \mathbf{G}_e}{\partial \hat{p}_e} \boldsymbol{\phi}_{i,e} + \mu_i \boldsymbol{\phi}_{i,e}^T \frac{\partial \mathbf{K}_e}{\partial \hat{p}_e} \boldsymbol{\phi}_{i,e} - \mathbf{w}_{i,e}^T \frac{\partial \mathbf{K}_e}{\partial \hat{p}_e} \mathbf{u}_e \right]
$$

dove il sistema aggiunto per ciascun modo $i = 1, \dots, n_{\text{eig}}$ è:

$$
\mathbf{K} \mathbf{w}_i = \sum_{e=1}^m E_G(\hat{p}_e) \left( \boldsymbol{\phi}_{i,e}^T \frac{\partial \mathbf{G}_e^0}{\partial \mathbf{u}_e} \boldsymbol{\phi}_{i,e} \right)^T
$$

**Ottimizzazione Computazionale (`sla.factorized`)**: Poiché la matrice $\mathbf{K}$ è identica per l'equilibrio statico ($\mathbf{K}\mathbf{u}=\mathbf{F}$) e per tutti gli $n_{\text{eig}}$ sistemi aggiunti, il solutore calcola la **fattorizzazione LU sparsa di $\mathbf{K}$ una sola volta per iterazione** e risolve i $1 + n_{\text{eig}}$ sistemi lineari per sola sostituzione avanti/indietro.

Il gradiente aggregato KS segue infine la regola della catena con pesi esponenziali normalizzati $w_{KS,i}$:

$$
\frac{\partial \mu_{KS}}{\partial \hat{p}_e} = \sum_{i=1}^{n_{\text{eig}}} w_{KS,i} \frac{\partial \mu_i}{\partial \hat{p}_e}, \qquad w_{KS,i} = \frac{\exp\left(\rho_{KS}(\mu_i - \mu_{\max})\right)}{\sum_{j=1}^{n_{\text{eig}}} \exp\left(\rho_{KS}(\mu_j - \mu_{\max})\right)}
$$

---

## 6. Ottimizzazione Multi-Obiettivo (Weighted Sum) e Regola della Catena Rigorosa

### 6.1 Scalarizzazione di Pareto con Normalizzazione Dinamica L1
Per bilanciare rigidezza flessionale ($C$) e stabilità al buckling ($\mu_{KS}$), il problema multi-obiettivo viene scalarizzato con peso di trade-off $\alpha \in [0, 0.95]$ (Marler & Arora, 2004, Teorema 3.1). Poiché $C$ e $\mu_{KS}$ hanno ordini di grandezza e unità fisiche differenti, i rispettivi gradienti vengono normalizzati ad ogni iterazione tramite la loro norma media $L_1$:

$$
s_{\text{total}, e} = (1 - \alpha) \frac{\partial C / \partial \hat{p}_e}{\frac{1}{m}\left\| \nabla_{\hat{\mathbf{p}}} C \right\|_1} + \alpha \frac{\partial \mu_{KS} / \partial \hat{p}_e}{\frac{1}{m}\left\| \nabla_{\hat{\mathbf{p}}} \mu_{KS} \right\|_1}
$$

### 6.2 Regola della Catena Completa (Obiettivo e Volume) e Bisezione OC Esatta
Poiché sia l'obiettivo $F(\hat{\mathbf{p}})$ sia il vincolo di volume fisico $V(\hat{\mathbf{p}}) = \frac{1}{m}\sum_e \hat{p}_e$ dipendono dalle densità proiettate $\hat{\mathbf{p}}(\tilde{\mathbf{x}}(\mathbf{x}))$, la derivata rispetto alle variabili di progetto matematiche $x_e$ viene calcolata applicando esattamente la **Chain Rule** della proiezione di Heaviside e del filtro spaziale:

$$
\frac{\partial \hat{p}_i}{\partial \tilde{x}_i} = \frac{\beta \left(1 - \tanh^2(\beta(\tilde{x}_i - \eta))\right)}{\tanh(\beta\eta) + \tanh(\beta(1-\eta))}
$$

$$
\frac{\partial F}{\partial x_e} = \sum_{i \in \mathcal{N}_e} \frac{H_{ie}}{H_{s,i}} \frac{\partial F}{\partial \hat{p}_i} \frac{\partial \hat{p}_i}{\partial \tilde{x}_i}, \qquad
\frac{\partial V}{\partial x_e} = \sum_{i \in \mathcal{N}_e} \frac{H_{ie}}{H_{s,i}} \frac{\partial \hat{p}_i}{\partial \tilde{x}_i}
$$

Nel ciclo di bisezione degli **Optimality Criteria (OC)**:
1. Il fattore di aggiornamento $B_e$ include la derivata non uniforme del volume $\partial V / \partial x_e$:
   $$
   B_e = \sqrt{\max\left(0, \, -\frac{\partial F / \partial x_e}{\lambda_L \, \max(10^{-12}, \partial V / \partial x_e)}\right)}
   $$
2. Ad ogni passo della bisezione su $\lambda_L$, per la proposta $\mathbf{x}_{\text{new}} = \text{clip}(\mathbf{x} \circ \mathbf{B}, \max(x_{\min}, \mathbf{x}-m), \min(1, \mathbf{x}+m))$ viene ricalcolata la **vera frazione di volume fisica** $\hat{\mathbf{p}}(\tilde{\mathbf{x}}(\mathbf{x}_{\text{new}}))$, garantendo il rispetto esatto del vincolo $\frac{1}{m}\sum_e \hat{p}_{e,\text{new}} = V_f$.

---

## 7. Ricostruzione delle Tensioni Principali ($\sigma_I, \sigma_{II}$) e Visualizzazione

Dal campo di spostamenti convergente $\mathbf{u}_e$, il tensore degli sforzi baricentrici $\boldsymbol{\sigma}_e = [\sigma_x, \sigma_y, \tau_{xy}]^T = \mathbf{C}_0 \mathbf{B}_0 \mathbf{u}_e$ permette di estrarre le tensioni principali nel piano:

$$
\sigma_{I,II} = \frac{\sigma_x + \sigma_y}{2} \pm \sqrt{\left(\frac{\sigma_x - \sigma_y}{2}\right)^2 + \tau_{xy}^2}
$$

In conformità con gli standard della letteratura sul buckling (Ferrari et al., 2021):
- Il post-processore maschera gli elementi di vuoto ($\hat{p}_e < \rho_{\text{thresh}}$) e utilizza la scala cromatica invertita **`Jet_r`**, mappando i valori fortemente negativi (**Compressione**, rischio buckling) in **Rosso** e i valori positivi (**Trazione**) in **Blu**.
- La funzione **Pareto Frontier Sweep** risolve automaticamente il problema per $\alpha \in \{0.0, 0.1, \dots, 1.0\}$ con contorni lisci (`smooth=True`), tracciando a posteriori il compromesso ottimo tra rigidezza ($C$) e stabilità ($\text{BLF} = \lambda_1$).

---

## 8. Riferimenti Bibliografici Principali

1. **Ferrari, F., Sigmund, O., & Guest, J. K. (2021).** *Topology optimization with linearized buckling criteria in 250 lines of Matlab.* Structural and Multidisciplinary Optimization, 63, 3045–3066.
2. **Ferrari, F., & Sigmund, O. (2020).** *Revisiting topology optimization with buckling constraints.* Structural and Multidisciplinary Optimization, 61(4), 1401–1415.
3. **Sigmund, O. (2001).** *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
4. **Wang, F., Lazarov, B. S., & Sigmund, O. (2011).** *On projection methods, convergence and robust formulations in topology optimization.* Structural and Multidisciplinary Optimization, 43(6), 767–784.
5. **Marler, R. T., & Arora, J. S. (2004).** *Survey of multi-objective optimization methods for engineering.* Structural and Multidisciplinary Optimization, 26(6), 369–395.
6. **Kreisselmeier, G., & Steinhauser, R. (1979).** *Systematic control design by optimizing a vector performance index.* IFAC Proceedings Volumes, 12(7), 113–117.
