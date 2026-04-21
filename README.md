# FL Backdoor Defense — FedAvg Baseline

**Course:** Federated Learning Research  
**Category:** Cat. 2 — Poisoning & Adversarial  
**Paper:** DifFense — Backdoor Defense in Federated Learning Using Differential Testing and Outlier Detection

---

## Project Structure

```
fl_project/
├── src/
│   ├── model.py          # CNN (MNIST/FMNIST) + ResNet-18 (CIFAR-10/100)
│   ├── data.py           # Dataset loading + Dirichlet Non-IID partitioning
│   ├── utils.py          # Training, evaluation, metrics, plotting
│   ├── client.py         # Flower ClientApp (FedAvg)
│   ├── server.py         # Flower ServerApp (FedAvg with server-side eval)
│   └── run_fedavg.py     # Main experiment runner
├── configs/              # YAML configs (coming in Stage 2)
├── results/              # Auto-generated: CSVs + plots
├── report/               # Final report PDF
└── requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running Experiments

### Quick single run
```bash
cd src

# CIFAR-10, 10 clients, 100 rounds, alpha=0.5
python run_fedavg.py --dataset cifar10 --num_clients 10 --num_rounds 100 --alpha 0.5

# FashionMNIST, 50 clients, IID
python run_fedavg.py --dataset fmnist --num_clients 50 --num_rounds 150 --alpha iid

# MNIST, 100 clients, very non-IID
python run_fedavg.py --dataset mnist --num_clients 100 --num_rounds 50 --alpha 0.01
```

### Run ALL required configurations
```bash
python run_fedavg.py --run_all
```
This runs all combinations of:
- Datasets: MNIST, FashionMNIST, CIFAR-10
- Clients: 10, 50, 100  
- Alpha: 0.01, 0.1, 0.5, 1.0, IID

---

## Configuration (Course-Mandated Settings)

| Parameter | Value |
|---|---|
| Client Fraction | 0.5 |
| Local Epochs | 5 |
| Batch Size | 32 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Learning Rate | 0.01 |
| Loss | Cross-Entropy |
| Seeds | 42 (all) |

---

## Output

Results are saved to `results/`:
- `fedavg_{dataset}_c{clients}_r{rounds}_a{alpha}.csv` — per-round metrics
- `figure4_iid_noniid_{dataset}_c{clients}.png` — IID vs Non-IID plots
- `fedavg_all_results.csv` — combined summary table
