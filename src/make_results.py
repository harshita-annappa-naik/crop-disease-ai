import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

central = json.load(open("results/central.json"))
iid = json.load(open("results/fedavg_iid.json"))
noniid = json.load(open("results/fedavg_noniid.json"))

rows = [("Centralised", central["test_accuracy"], "-"),
        ("FedAvg IID", iid["final_test_accuracy"], iid["total_mb_sent"]),
        ("FedAvg non-IID", noniid["final_test_accuracy"], noniid["total_mb_sent"])]
print(f"{'Method':<18}{'Test acc':>10}{'MB sent':>10}")
for name, acc, mb in rows:
    print(f"{name:<18}{acc:>10.4f}{str(mb):>10}")

plt.figure(figsize=(6, 4))
for label, res in [("IID", iid), ("non-IID", noniid)]:
    plt.plot([r["round"] for r in res["rounds"]],
             [r["test_accuracy"] for r in res["rounds"]], marker="o", label=label)
plt.axhline(central["test_accuracy"], ls="--", color="gray", label="Centralised")
plt.xlabel("Round")
plt.ylabel("Test accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/accuracy_per_round.png", dpi=120)
print("Saved results/accuracy_per_round.png")