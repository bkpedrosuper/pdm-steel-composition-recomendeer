"""Promove uma versão a champion, com gate de qualidade.

Compara o MAE de validação cruzada (geral, por saída) da candidata com o do champion atual.
Se a candidata piorar mais que a tolerância em alguma saída, a promoção é recusada
(a não ser com --force). A API de produção serve `models:/pdm-surrogate@champion`, então
promover = trocar o alias; rollback = promover a versão anterior.

Uso:
    python -m pdm.promote 5                    # champion -> v5, se passar no gate
    python -m pdm.promote 5 --dry-run          # só compara, não muda nada
    python -m pdm.promote 4 --force            # rollback/forçar, ignora o gate
    python -m pdm.promote 5 --alias challenger # só marca a candidata, sem gate
"""
import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from pdm.artifacts import MODEL_NAME, tracking_uri

TARGETS = ["LE", "LR", "AL"]


def cv_mae(client: MlflowClient, version: str) -> dict[str, float]:
    """MAE de validação cruzada (todos os aços) da versão, por saída."""
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    metrics = client.get_run(run_id).data.metrics if run_id else {}
    return {t: metrics[f"cv_ALL_{t}_mae"] for t in TARGETS if f"cv_ALL_{t}_mae" in metrics}


def current_champion(client: MlflowClient) -> str | None:
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
    except MlflowException:
        return None   # ainda não há champion: a primeira versão entra direto


def gate(client: MlflowClient, candidate: str, champion: str, tolerance: float) -> bool:
    new, old = cv_mae(client, candidate), cv_mae(client, champion)
    ok = True
    print(f"{'saída':<6}{'champion v' + champion:>16}{'candidata v' + candidate:>16}{'variação':>11}")
    for t in TARGETS:
        if t not in new or t not in old:
            print(f"{t:<6} sem métrica para comparar"); ok = False; continue
        change = (new[t] - old[t]) / old[t]
        worse = change > tolerance
        ok &= not worse
        print(f"{t:<6}{old[t]:>16.3f}{new[t]:>16.3f}{change:>+10.1%}{'  <- piorou' if worse else ''}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="número da versão no Model Registry")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="piora relativa de MAE aceita por saída (padrão 2%%)")
    parser.add_argument("--force", action="store_true", help="ignora o gate (rollback)")
    parser.add_argument("--dry-run", action="store_true", help="só compara, não muda o alias")
    args = parser.parse_args()

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    version = str(args.version)

    if args.alias == "champion":
        champion = current_champion(client)
        if champion and champion != version:
            print(f"Gate: MAE de validação cruzada, tolerância {args.tolerance:.0%}\n")
            passed = gate(client, version, champion, args.tolerance)
            if not passed and not args.force:
                raise SystemExit(f"\nREPROVADA: v{version} não substitui o champion v{champion}. "
                                 "Use --force se for intencional (ex.: rollback).")
            print("\n" + ("APROVADA" if passed else "FORÇADA (gate ignorado)"))
        if args.dry_run:
            print("--dry-run: nenhum alias alterado")
            return

    client.set_registered_model_alias(MODEL_NAME, args.alias, version)
    print()
    for v in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
        aliases = " ".join(f"@{a}" for a in (v.aliases or []))
        print(f"versão {v.version} (run {(v.run_id or '')[:8]}) {aliases}")


if __name__ == "__main__":
    main()
