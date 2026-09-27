# Regole di Progetto & Standard Operativi Permanenti (Workspace Rules)

Questo file definisce le direttive permanenti per il progetto **Computational Engineering** (PicoGK Dashboard, Solutori FEA e Generative Design).

---

## 1. Fondamenti Matematici & Risolutori Fisici (FEA & Generative Core)
- **Unità di Misura Coerenti**: Utilizzare sempre il sistema coerente per la meccanica strutturale:
  - Lunghezze in millimetri ($\text{mm}$)
  - Forze in Newton ($\text{N}$)
  - Tensioni e Modulo di Young in MegaPascal ($\text{MPa} = \text{N/mm}^2$)
  - Spostamenti in millimetri ($\text{mm}$)
- **Validazione Analitica Rigorosa**:
  - Ogni nuovo elemento finito (es. Q4, T3, H8) o solutore deve essere validato contro casi teorici a soluzione chiusa (es. mensola a sbalzo di De Saint-Venant / Eulero-Bernoulli).
  - La differenza rispetto alla soluzione analitica deve essere tracciata e documentata.
- **Risolutori Sparsi & Efficienza**:
  - Utilizzare matrici sparse (`scipy.sparse` in formato CSC o CSR).
  - Pre-calcolare le matrici elementari $\mathbf{k}_0$ sulle griglie regolari senza ricalcoli ridondanti.

---

## 2. Dashboard Streamlit & Visualizzazione (`building-data-apps` & `generative_ui`)
- **Interattività Real-Time**:
  - Mantenere la UI fluida e reattiva; i tempi di calcolo per mesh 2D devono restare sotto i 100 ms.
  - Fornire slider con range fisici sensati (es. fattore di amplificazione deformata, soglia di snervamento von Mises).
- **Mappe di Tensione Chiare**:
  - Le scale di colore (colormap) devono indicare chiaramente la soglia di snervamento $\sigma_{yield}$ con barra graduata leggibile.
- **Visualizzazione 3D**:
  - Utilizzare Three.js / WebGL / PyVista per il rendering geometrico di file `.stl` e `.obj` generati da PicoGK/OpenVDB.

---

## 3. Gestione Dipendenze & Ambiente Python (`managing-python-dependencies`)
- Utilizzare sempre l'ambiente virtuale dedicato al progetto per evitare conflitti con pacchetti globali.
- Tracciare ogni nuova dipendenza scientifica (`numpy`, `scipy`, `streamlit`, `pyvista`, `pydantic`) nel file `requirements.txt`.
