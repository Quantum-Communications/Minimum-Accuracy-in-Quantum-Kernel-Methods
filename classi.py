import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_circles, make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import warnings
import os
import datetime
import glob

warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp


N_QUBITS = 6                     
N_SAMPLES = 1000                 
N_TRAIN = 200                    
BASE_RANDOM_STATE = 42           
N_REPETITIONS = 30               
RUN_DETERMINISTIC = True         
MAX_D_FOR_PAULI_SVM = 4096       
SCALE_FACTOR = np.pi             
INCLUDE_CLASSICAL_BASELINES = True
N_LAYERS = 2                     


FEATURE_MAP_TYPE = "RY_CRZ"      


RANDOM_FEATURE_SEED = 42

def create_output_folder(base_path="experimentos"):
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    pattern = os.path.join(base_path, f"{today}_*")
    existing = glob.glob(pattern)
    
    if not existing:
        seq = 1
    else:
        numbers = []
        for folder in existing:
            basename = os.path.basename(folder)
            try:
                num = int(basename.split('_')[-1])
                numbers.append(num)
            except:
                pass
        seq = max(numbers) + 1 if numbers else 1
    folder_name = f"{today}_{seq:03d}"
    full_path = os.path.join(base_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

OUTPUT_DIR = create_output_folder()
print(f"Resultados serão salvos em: {OUTPUT_DIR}")


with open(os.path.join(OUTPUT_DIR, "README.txt"), "w") as f:
    f.write("=== Config ===\n")
    f.write(f"Data e hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"FEATURE_MAP_TYPE: {FEATURE_MAP_TYPE}\n")
    f.write(f"N_QUBITS: {N_QUBITS} (d = {4**N_QUBITS})\n")
    f.write(f"N_SAMPLES: {N_SAMPLES}\n")
    f.write(f"N_TRAIN: {N_TRAIN}\n")
    f.write(f"N_REPETITIONS: {N_REPETITIONS}\n")
    f.write(f"RUN_DETERMINISTIC: {RUN_DETERMINISTIC}\n")
    f.write(f"MAX_D_FOR_PAULI_SVM: {MAX_D_FOR_PAULI_SVM}\n")
    f.write(f"SCALE_FACTOR: {SCALE_FACTOR}\n")
    f.write(f"INCLUDE_CLASSICAL_BASELINES: {INCLUDE_CLASSICAL_BASELINES}\n")
    f.write(f"N_LAYERS: {N_LAYERS}\n")
    f.write(f"RANDOM_FEATURE_SEED: {RANDOM_FEATURE_SEED}\n")
    f.write(f"BASE_RANDOM_STATE: {BASE_RANDOM_STATE}\n")
    f.write("================================\n")


PAULI_MAP = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

def idx_to_pauli_string(idx: int, n_qubits: int) -> str:
    digits = []
    for _ in range(n_qubits):
        idx, r = divmod(idx, 4)
        digits.append(PAULI_MAP[r])
    digits.reverse()
    return ''.join(digits)

def build_feature_map_RY_CRZ(x: np.ndarray, n_qubits: int, n_layers: int = 1) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, 'q')
    circuit = QuantumCircuit(qr)
    for _ in range(n_layers):
        for i in range(min(n_qubits, len(x))):
            angle = SCALE_FACTOR * x[i]
            circuit.ry(angle, qr[i])
        for i in range(min(n_qubits-1, len(x)-1)):
            angle = SCALE_FACTOR * x[i]
            circuit.crz(angle, qr[i], qr[i+1])
    circuit.h(qr)
    return circuit


def build_feature_map_ZZ(x: np.ndarray, n_qubits: int, n_layers: int = 1) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, 'q')
    circuit = QuantumCircuit(qr)
    for _ in range(n_layers):
        for i in range(min(n_qubits, len(x))):
            angle = SCALE_FACTOR * x[i]
            circuit.rz(angle, qr[i])
        for i in range(min(n_qubits-1, len(x)-1)):
            angle = SCALE_FACTOR * (x[i] * x[i+1])
            circuit.rzz(angle, qr[i], qr[i+1])
    circuit.h(qr)
    return circuit

def build_feature_map_RANDOM(x: np.ndarray, n_qubits: int, n_ops: int = None) -> QuantumCircuit:
    if n_ops is None:
        n_ops = max(2, n_qubits)
    qr = QuantumRegister(n_qubits, 'q')
    circuit = QuantumCircuit(qr)
    rng = np.random.default_rng(RANDOM_FEATURE_SEED)
    qubit_pairs = [tuple(rng.choice(n_qubits, size=2, replace=False)) for _ in range(n_ops)]
    feature_pairs = [tuple(rng.choice(len(x), size=2, replace=False)) for _ in range(n_ops)]
    for (q1, q2), (f1, f2) in zip(qubit_pairs, feature_pairs):
        angle = SCALE_FACTOR * (x[f1] * x[f2])
        circuit.ryy(angle, q1, q2)
    circuit.h(qr)
    return circuit

def build_quantum_feature_map(x: np.ndarray, n_qubits: int) -> QuantumCircuit:
    if FEATURE_MAP_TYPE == "RY_CRZ":
        return build_feature_map_RY_CRZ(x, n_qubits, N_LAYERS)
    elif FEATURE_MAP_TYPE == "ZZ":
        return build_feature_map_ZZ(x, n_qubits, N_LAYERS)
    elif FEATURE_MAP_TYPE == "RANDOM":
        return build_feature_map_RANDOM(x, n_qubits)
    else:
        raise ValueError(f"Tipo de feature map desconhecido: {FEATURE_MAP_TYPE}")

def get_statevector(x: np.ndarray, n_qubits: int) -> Statevector:
    circuit = build_quantum_feature_map(x, n_qubits)
    return Statevector.from_instruction(circuit)

def compute_expectation(sv: Statevector, pauli_str: str) -> float:
    op = Pauli(pauli_str)
    return sv.expectation_value(op).real


class QuantumPauliProjector:
    def __init__(self, X: np.ndarray, n_qubits: int):
        self.X = X
        self.n_qubits = n_qubits
        self.N = X.shape[0]
        print(f"    [Projetor] Inicializado com {self.N} amostras, {n_qubits} qubits")

    def get_projections(self, pauli_str: str) -> np.ndarray:
        proj = np.zeros(self.N)
        for i in range(self.N):
            sv = get_statevector(self.X[i], self.n_qubits)
            proj[i] = compute_expectation(sv, pauli_str)
        return proj

def rmin_on_axis(projections: np.ndarray, y: np.ndarray) -> float:
    N = len(y)
    order = np.argsort(projections)
    proj_sorted = projections[order]
    y_sorted = y[order]
    P_pos = np.zeros(N+1, dtype=int)
    P_neg = np.zeros(N+1, dtype=int)
    for t in range(1, N+1):
        P_pos[t] = P_pos[t-1] + (1 if y_sorted[t-1]==1 else 0)
        P_neg[t] = P_neg[t-1] + (1 if y_sorted[t-1]==-1 else 0)
    total_pos = P_pos[N]
    total_neg = P_neg[N]
    best = 0.0
    for t in range(N+1):
        left_pos = P_pos[t]
        left_neg = P_neg[t]
        right_pos = total_pos - left_pos
        right_neg = total_neg - left_neg
        acc1 = (left_pos + right_neg)/N
        acc2 = (left_neg + right_pos)/N
        best = max(best, acc1, acc2)
    return best

class MinimumAccuracyEvaluatorQuantum:
    def __init__(self, projector: QuantumPauliProjector, y: np.ndarray):
        self.projector = projector
        self.N, self.d = projector.N, 4**projector.n_qubits
        y = np.array(y)
        if set(np.unique(y)) == {0,1}:
            y = np.where(y==0, -1, 1)
        self.y = y.astype(int)
        print(f"    [Avaliador] d={self.d}, N={self.N}")

    def _evaluate_axis(self, idx):
        pauli_str = idx_to_pauli_string(idx, self.projector.n_qubits)
        proj = self.projector.get_projections(pauli_str)
        return rmin_on_axis(proj, self.y)

    def deterministic(self):
        print(f"    [Deterministic] Iniciando varredura de {self.d} eixos...")
        start = time.time()
        best = 0.0
        best_axis = -1
        for i in range(self.d):
            acc = self._evaluate_axis(i)
            if acc > best:
                best = acc
                best_axis = i
            if (i+1) % 500 == 0:
                print(f"      Eixos processados: {i+1}/{self.d}, melhor acurácia até agora: {best:.4f}")
        elapsed = time.time() - start
        print(f"    [Deterministic] Concluído. Melhor acurácia: {best:.4f}, eixo: {best_axis}, tempo: {elapsed:.2f}s")
        return {"accuracy": best, "best_axis": best_axis, "n_axes": self.d, "time": elapsed}

    def conservative(self, p=0.05, delta=0.05):
        t = int(np.ceil((1.0/p)*np.log(1.0/delta)))
        t = min(t, self.d)
        print(f"    [Conservative] Amostrando {t} eixos (p={p}, delta={delta})...")
        start = time.time()
        axes = np.random.choice(self.d, size=t, replace=False)
        best = 0.0
        for i, ax in enumerate(axes):
            acc = self._evaluate_axis(ax)
            if acc > best:
                best = acc
            if (i+1) % 20 == 0:
                print(f"      Amostras processadas: {i+1}/{t}, melhor acurácia: {best:.4f}")
        elapsed = time.time() - start
        print(f"    [Conservative] Concluído. Melhor acurácia: {best:.4f}, tempo: {elapsed:.2f}s")
        return {"accuracy": best, "n_axes": t, "time": elapsed}

    def pilot(self, n_pilot=50, delta=0.05, pilot_fraction=0.005, max_fraction=0.15, p_floor=0.01, tau_percentile=95):
        print(f"    [Pilot] Estágio piloto com {n_pilot} eixos...")
        start = time.time()
        pilot_by_fraction = max(int(np.ceil(pilot_fraction * self.d)), 50)
        adaptive_pilot = max(n_pilot, pilot_by_fraction)
        adaptive_pilot = min(adaptive_pilot, self.d)
        adaptive_pilot = max(adaptive_pilot, min(100, self.d))
        pilot_axes = np.random.choice(self.d, size=adaptive_pilot, replace=False)
        pilot_results = []
        for i, ax in enumerate(pilot_axes):
            acc = self._evaluate_axis(ax)
            pilot_results.append((ax, acc))
            if (i+1) % 20 == 0:
                print(f"      Piloto: {i+1}/{adaptive_pilot}, última acurácia: {acc:.4f}")
        pilot_accs = np.array([r[1] for r in pilot_results])
        acc_mean = np.mean(pilot_accs)
        acc_std = np.std(pilot_accs)
        tau = np.percentile(pilot_accs, tau_percentile)
        p_emp = np.mean(pilot_accs >= tau)
        p_hat = max(p_emp, p_floor)
        print(f"      Piloto: média={acc_mean:.4f}, std={acc_std:.4f}, tau={tau:.4f}, p_hat={p_hat:.4f}")
        t_base = (1.0/p_hat) * np.log(1.0/delta)
        uncertainty_factor = 1.0 + (acc_std/0.2)
        uncertainty_factor = min(uncertainty_factor, 2.5)
        t_adjusted = t_base * uncertainty_factor
        random_factor = np.random.uniform(0.8,1.2)
        t_required = int(np.ceil(t_adjusted * random_factor))
        t_max = int(np.ceil(max_fraction * self.d))
        t_max = max(t_max, adaptive_pilot)
        t_required = min(t_required, t_max)
        min_additional = int(adaptive_pilot * (0.5+acc_std))
        t_required = max(t_required, adaptive_pilot + min_additional)
        print(f"      Tamanho total requerido: {t_required} eixos")
        all_results = pilot_results.copy()
        if t_required > adaptive_pilot:
            remaining = list(set(range(self.d)) - set(pilot_axes))
            n_additional = min(t_required - adaptive_pilot, len(remaining))
            print(f"      Amostrando mais {n_additional} eixos...")
            if n_additional > 0:
                additional_axes = np.random.choice(remaining, size=n_additional, replace=False)
                for i, ax in enumerate(additional_axes):
                    acc = self._evaluate_axis(ax)
                    all_results.append((ax, acc))
                    if (i+1) % 20 == 0:
                        print(f"        Adicionais: {i+1}/{n_additional}, última acurácia: {acc:.4f}")
        best_axis, best_acc = max(all_results, key=lambda x: x[1])
        elapsed = time.time() - start
        print(f"    [Pilot] Concluído. Melhor acurácia: {best_acc:.4f}, eixos usados: {len(all_results)}, tempo: {elapsed:.2f}s")
        return {"accuracy": best_acc, "n_axes": len(all_results), "time": elapsed}

    def adaptive(self, batch_size=30, max_axes=None, convergence_threshold=1e-3, patience=3):
        print(f"    [Adaptive] Iniciando busca adaptativa...")
        start = time.time()
        if max_axes is None:
            max_axes = min(500, int(0.1*self.d))
        adaptive_batch = min(batch_size, max(int(0.005*self.d),10))
        sampled = set()
        best_acc = 0.0
        recent_best = []
        no_improvement = 0
        total_processed = 0
        while len(sampled) < min(max_axes, self.d):
            remaining = list(set(range(self.d)) - sampled)
            if not remaining: break
            current_batch_size = min(adaptive_batch, len(remaining))
            batch = np.random.choice(remaining, size=current_batch_size, replace=False)
            batch_best = best_acc
            for i, ax in enumerate(batch):
                sampled.add(ax)
                acc = self._evaluate_axis(ax)
                if acc > best_acc:
                    best_acc = acc
                total_processed += 1
            recent_best.append(best_acc)
            if best_acc > batch_best:
                no_improvement = 0
            else:
                no_improvement += 1
            print(f"      Lote {len(recent_best)}: processados {total_processed} eixos, melhor acurácia={best_acc:.4f}, paciência={no_improvement}/{patience}")
            if no_improvement >= patience:
                print(f"        Parando por paciência (nenhuma melhoria em {patience} lotes).")
                break
            if len(recent_best) >= 5 and np.std(recent_best[-5:]) < convergence_threshold:
                print(f"        Parando por estabilidade (desvio padrão < {convergence_threshold}).")
                break
        elapsed = time.time() - start
        print(f"    [Adaptive] Concluído. Melhor acurácia: {best_acc:.4f}, eixos usados: {len(sampled)}, tempo: {elapsed:.2f}s")
        return {"accuracy": best_acc, "n_axes": len(sampled), "time": elapsed}

def generate_datasets(n_samples=1000, random_state=42):
    datasets = {}
    rs = random_state
    X, y = make_blobs(n_samples=n_samples, n_features=4, centers=2, cluster_std=1.0, random_state=rs)
    y = (y==1).astype(int)
    datasets['Blobs'] = (X,y)
    datasets['Circles'] = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=rs)
    datasets['Linear_Separable'] = make_classification(n_samples=n_samples, n_features=4, n_redundant=0,
                                                       n_informative=4, n_clusters_per_class=1, flip_y=0.0, random_state=rs)
    datasets['Moons'] = make_moons(n_samples=n_samples, noise=0.1, random_state=rs)
    datasets['Multi_Cluster'] = make_classification(n_samples=n_samples, n_features=4, n_redundant=0,
                                                    n_informative=4, n_clusters_per_class=3, random_state=rs)
    return datasets

def train_linear_svm_pauli_space_optimized(projector: QuantumPauliProjector, y_train: np.ndarray, random_state: int) -> float:
    n_qubits = projector.n_qubits
    N = projector.N
    d = 4 ** n_qubits
    print(f"    [SVM Pauli-Otimizado] Construindo matriz Phi ({N} x {d})...")
    start_time = time.time()
    # Cache statevectors
    print("      Calculando statevectors para todas as amostras...")
    statevectors = [get_statevector(projector.X[i], n_qubits) for i in range(N)]
    print("      Gerando strings de Pauli...")
    pauli_strings = [idx_to_pauli_string(i, n_qubits) for i in range(d)]
    print("      Calculando expectativas em lote...")
    pauli_ops = [SparsePauliOp(ps) for ps in pauli_strings]
    Phi = np.zeros((N, d), dtype=np.float64)
    for j, sv in enumerate(statevectors):
        if j % 20 == 0:
            print(f"        Processando amostra {j+1}/{N}...")
        for i, op in enumerate(pauli_ops):
            Phi[j, i] = sv.expectation_value(op).real
    print(f"      Matriz Phi construída em {time.time() - start_time:.2f}s")
    col_var = np.var(Phi, axis=0)
    n_zero_var = np.sum(col_var < 1e-10)
    print(f"    [SVM Pauli-Otimizado] Colunas com variância zero: {n_zero_var}/{d}")
    if n_zero_var == d:
        print("    [ERRO] Todas as projeções são constantes! Verifique o feature map.")
        return 0.5
    scaler = StandardScaler()
    Phi_scaled = scaler.fit_transform(Phi)
    svm = LinearSVC(C=100.0, dual=True, max_iter=100000, tol=1e-5,
                    random_state=random_state, loss='squared_hinge')
    svm.fit(Phi_scaled, y_train)
    y_pred = svm.predict(Phi_scaled)
    acc = accuracy_score(y_train, y_pred)
    print(f"    [SVM Pauli-Otimizado] Acurácia no treino: {acc:.4f}")
    return acc

train_linear_svm_pauli_space = train_linear_svm_pauli_space_optimized


@dataclass
class ExperimentResult:
    dataset_name: str
    method: str
    accuracy: float
    time_elapsed: float
    n_axes_sampled: int = None

def run_experiment(dataset_name, X, y, n_qubits, random_state):
    print(f"\n  --- Processando dataset: {dataset_name} (seed={random_state}) ---")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_use = X_train_scaled[:N_TRAIN]
    y_train_use = y_train[:N_TRAIN]
    print(f"  Treinamento com {len(X_train_use)} amostras, {n_qubits} qubits")
    projector = QuantumPauliProjector(X_train_use, n_qubits)
    results = []

    if INCLUDE_CLASSICAL_BASELINES:
        print("  SVM Linear (features originais)...")
        svm_lin = SVC(kernel='linear', random_state=random_state).fit(X_train_use, y_train_use)
        acc_lin = accuracy_score(y_train_use, svm_lin.predict(X_train_use))
        results.append(ExperimentResult(dataset_name, "SVM_Linear_OriginalFeatures", acc_lin, 0.0))
        print(f"    Acurácia: {acc_lin:.4f}")
        print("  SVM RBF (features originais)...")
        svm_rbf = SVC(kernel='rbf', random_state=random_state).fit(X_train_use, y_train_use)
        acc_rbf = accuracy_score(y_train_use, svm_rbf.predict(X_train_use))
        results.append(ExperimentResult(dataset_name, "SVM_RBF_OriginalFeatures", acc_rbf, 0.0))
        print(f"    Acurácia: {acc_rbf:.4f}")

    d = 4**n_qubits
    if d <= MAX_D_FOR_PAULI_SVM:
        acc_pauli = train_linear_svm_pauli_space(projector, y_train_use, random_state)
        results.append(ExperimentResult(dataset_name, "SVM_Linear_PauliSpace", acc_pauli, 0.0))

    evaluator = MinimumAccuracyEvaluatorQuantum(projector, y_train_use)

    if RUN_DETERMINISTIC:
        det = evaluator.deterministic()
        results.append(ExperimentResult(dataset_name, "Rmin_Deterministic", det['accuracy'], det['time'], det['n_axes']))

    cons = evaluator.conservative()
    results.append(ExperimentResult(dataset_name, "Rmin_Conservative", cons['accuracy'], cons['time'], cons['n_axes']))

    pilot = evaluator.pilot()
    results.append(ExperimentResult(dataset_name, "Rmin_Pilot", pilot['accuracy'], pilot['time'], pilot['n_axes']))

    adapt = evaluator.adaptive()
    results.append(ExperimentResult(dataset_name, "Rmin_Adaptive", adapt['accuracy'], adapt['time'], adapt['n_axes']))

    return results

def run_multi_repetitions():
    all_records = []
    for rep in range(N_REPETITIONS):
        print(f"\n{'='*60}\nRepetition {rep+1}/{N_REPETITIONS}\n{'='*60}")
        current_seed = BASE_RANDOM_STATE + rep
        datasets = generate_datasets(n_samples=N_SAMPLES, random_state=current_seed)
        for name, (X, y) in datasets.items():
            res_list = run_experiment(name, X, y, N_QUBITS, current_seed)
            for res in res_list:
                all_records.append({
                    "rep": rep,
                    "dataset": res.dataset_name,
                    "method": res.method,
                    "accuracy": res.accuracy,
                    "time": res.time_elapsed,
                    "axes": res.n_axes_sampled if res.n_axes_sampled is not None else 0
                })
    df = pd.DataFrame(all_records)
    summary = df.groupby(["dataset", "method"]).agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        axes_mean=("axes", "mean"),
        axes_std=("axes", "std")
    ).reset_index()
    return df, summary

def plot_publication_quality(summary_df, n_qubits, output_dir):
    method_names = {
        "SVM_Linear_OriginalFeatures": "SVM Linear (orig. features)",
        "SVM_RBF_OriginalFeatures": "SVM RBF (orig. features)",
        "SVM_Linear_PauliSpace": "SVM Linear (Pauli space)",
        "Rmin_Deterministic": "Rmin Deterministic",
        "Rmin_Conservative": "Rmin Conservative",
        "Rmin_Pilot": "Rmin Pilot",
        "Rmin_Adaptive": "Rmin Adaptive"
    }
    summary_df["Method"] = summary_df["method"].map(method_names)
    order_full = ["SVM Linear (Pauli space)", "SVM Linear (orig. features)", "SVM RBF (orig. features)",
                  "Rmin Deterministic", "Rmin Conservative", "Rmin Pilot", "Rmin Adaptive"]
    available_methods = [m for m in order_full if m in summary_df["Method"].unique()]
    summary_df = summary_df[summary_df["Method"].isin(available_methods)]
    summary_df["Method"] = pd.Categorical(summary_df["Method"], categories=available_methods, ordered=True)
    summary_df = summary_df.sort_values(["dataset", "Method"])
    datasets = summary_df["dataset"].unique()
    palette = sns.color_palette("Set2", n_colors=len(available_methods))
    color_dict = {meth: palette[i] for i, meth in enumerate(available_methods)}
    
    # Acurácia
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(datasets))
    width = 0.12
    for j, meth in enumerate(available_methods):
        subset = summary_df[summary_df["Method"] == meth]
        means = [subset[subset["dataset"]==d]["acc_mean"].values[0] if not subset[subset["dataset"]==d].empty else 0 for d in datasets]
        stds = [subset[subset["dataset"]==d]["acc_std"].values[0] if not subset[subset["dataset"]==d].empty else 0 for d in datasets]
        offset = width * (j - len(available_methods)/2 + 0.5)
        ax.bar(x+offset, means, width, label=meth, color=color_dict[meth], yerr=stds, capsize=3, error_kw={'linewidth':1.5})
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel("Training Accuracy")
    ax.set_title(f"Training Accuracy (mean ± std) over {N_REPETITIONS} runs\n({n_qubits} qubits, {FEATURE_MAP_TYPE})")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1)); ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"accuracy_{n_qubits}qubits_{FEATURE_MAP_TYPE}.png"), dpi=300, bbox_inches='tight'); plt.close()
    
    # Eixos amostrados
    rmin_methods = [m for m in available_methods if "Rmin" in m]
    if rmin_methods:
        df_rmin = summary_df[summary_df["Method"].isin(rmin_methods)]
        fig, ax = plt.subplots(figsize=(10,5))
        x = np.arange(len(datasets))
        width = 0.2
        for j, meth in enumerate(rmin_methods):
            subset = df_rmin[df_rmin["Method"] == meth]
            means = [subset[subset["dataset"]==d]["axes_mean"].values[0] if not subset[subset["dataset"]==d].empty else 0 for d in datasets]
            stds = [subset[subset["dataset"]==d]["axes_std"].values[0] if not subset[subset["dataset"]==d].empty else 0 for d in datasets]
            offset = width * (j - len(rmin_methods)/2 + 0.5)
            ax.bar(x+offset, means, width, label=meth, yerr=stds, capsize=3, error_kw={'linewidth':1.5}, color=color_dict[meth])
        ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel("Number of Pauli axes sampled")
        ax.set_title(f"Axes sampled by Rmin methods (mean ± std)\n({n_qubits} qubits, {FEATURE_MAP_TYPE})")
        ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"axes_{n_qubits}qubits_{FEATURE_MAP_TYPE}.png"), dpi=300, bbox_inches='tight'); plt.close()
    
    # Tempo
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(datasets))
    width = 0.12
    for j, meth in enumerate(available_methods):
        subset = summary_df[summary_df["Method"] == meth]
        means = [subset[subset["dataset"]==d]["time_mean"].values[0] if not subset[subset["dataset"]==d].empty else 0 for d in datasets]
        stds = [subset[subset["dataset"]==d]["time_std"].values[0] if not subset[subset["dataset"]==d].empty else 0 for d in datasets]
        offset = width * (j - len(available_methods)/2 + 0.5)
        ax.bar(x+offset, means, width, label=meth, color=color_dict[meth], yerr=stds, capsize=3, error_kw={'linewidth':1.5})
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel("Execution time (seconds)")
    ax.set_title(f"Execution time (mean ± std) over {N_REPETITIONS} runs\n({n_qubits} qubits, {FEATURE_MAP_TYPE})")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1)); ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"time_{n_qubits}qubits_{FEATURE_MAP_TYPE}.png"), dpi=300, bbox_inches='tight'); plt.close()
    
    # Eficiência
    if rmin_methods:
        fig, ax = plt.subplots(figsize=(10,6))
        for meth in rmin_methods:
            subset = df_rmin[df_rmin["Method"] == meth]
            x_means = subset["axes_mean"].values
            x_stds = subset["axes_std"].values
            y_means = subset["acc_mean"].values
            y_stds = subset["acc_std"].values
            ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt='o', capsize=4, label=meth, markersize=8, elinewidth=1.5, capthick=1.5)
        if "Rmin Deterministic" in rmin_methods:
            det_axes = summary_df[summary_df["Method"] == "Rmin Deterministic"]["axes_mean"].mean()
            ax.axvline(x=det_axes, linestyle='--', color='gray', alpha=0.5, label=f'Deterministic ({int(det_axes)} axes)')
        ax.set_xlabel("Number of axes sampled")
        ax.set_ylabel("Rmin accuracy")
        ax.set_title(f"Efficiency trade-off: axes vs. accuracy (mean ± std)\n({n_qubits} qubits, {FEATURE_MAP_TYPE})")
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"efficiency_{n_qubits}qubits_{FEATURE_MAP_TYPE}.png"), dpi=300, bbox_inches='tight'); plt.close()
    
    print(f"All publication-ready plots saved in {output_dir}")

def main():
    print("="*70)
    print(f"Quantum Feature Map Evaluation – Type: {FEATURE_MAP_TYPE} (optimized)")
    print(f"Qubits: {N_QUBITS} -> d = {4**N_QUBITS}")
    print(f"Training samples per dataset: {N_TRAIN}")
    print(f"Repetitions: {N_REPETITIONS}")
    print(f"Deterministic method: {'ENABLED' if RUN_DETERMINISTIC else 'DISABLED'}")
    if 4**N_QUBITS <= MAX_D_FOR_PAULI_SVM:
        print("Pauli-space SVM baseline: ENABLED (d <= {})".format(MAX_D_FOR_PAULI_SVM))
    else:
        print("Pauli-space SVM baseline: DISABLED (d > {})".format(MAX_D_FOR_PAULI_SVM))
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print("="*70)

    
    print("\n=== feature map (diagnóstico) ===")
    n_test = 2
    X_test = np.random.randn(3, 4)
    op_test = Pauli('Z' + 'I'*(n_test-1))
    for idx, x in enumerate(X_test):
        sv = get_statevector(x, n_test)
        exp = sv.expectation_value(op_test).real
        print(f"Amostra {idx}: x[0]={x[0]:.4f}, exp(ZI)={exp:.6f}")

    raw_df, summary = run_multi_repetitions()

    
    summary.to_csv(os.path.join(OUTPUT_DIR, f"summary_{N_QUBITS}qubits_{FEATURE_MAP_TYPE}.csv"), index=False)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, f"raw_results_{N_QUBITS}qubits_{FEATURE_MAP_TYPE}.csv"), index=False)

    pd.set_option('display.max_columns', None)
    print("\n=== Summary (mean ± std) ===")
    print(summary[["dataset", "method", "acc_mean", "acc_std", "axes_mean", "axes_std", "time_mean"]])

    plot_publication_quality(summary, N_QUBITS, OUTPUT_DIR)
    print(f"\nExperiment finished successfully. Todos os resultados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
