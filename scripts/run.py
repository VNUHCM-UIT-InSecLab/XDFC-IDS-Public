from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explainability.shap_explainer import LocalSHAPAnalyzer
from federated.dfl_trainer import DecentralizedFederatedTrainer, XDFCIDSConfig
from models.factory import build_model
from utils.data import (
    load_data,
    parse_int_schedule
)
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XDFC-IDS")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=9)

    parser.add_argument("--model", choices=["cnn_gru", "lstm"], default="cnn_gru")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--client-schedule", type=str, default="30,40,50")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--retention-ratio", type=float, default=0.60)
    parser.add_argument("--ewc-lambda", type=float, default=1e-3)
    parser.add_argument("--fisher-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lstm-units", type=int, default=50)
    parser.add_argument("--lstm-hidden-units", type=int, default=0)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="runs/xdfc_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_schedule = parse_int_schedule(args.client_schedule)

    if not args.data_path:
        raise ValueError("data-path is required")

    task_partitions, test_sets, feature_names, class_names = load_data(
        path=args.data_path,
        alpha=args.alpha,
    )

    input_shape = task_partitions[0][0][0].shape[1:]

    def model_builder():
        kwargs: Dict = {}
        if args.model == "lstm":
            kwargs["lstm_units"] = args.lstm_units
            kwargs["hidden_units"] = args.lstm_hidden_units if args.lstm_hidden_units > 0 else None
        return build_model(args.model, input_shape=input_shape, num_classes=args.num_classes, **kwargs)

    config = XDFCIDSConfig(
        num_collaborators=client_schedule[0],
        communication_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        retention_ratio=args.retention_ratio,
        ewc_lambda=args.ewc_lambda,
        fisher_batches=args.fisher_batches,
        seed=args.seed,
    )
    trainer = DecentralizedFederatedTrainer(model_builder=model_builder, config=config)
    analyzer = LocalSHAPAnalyzer(top_k=args.top_k) if args.explain else None

    history: List[Dict] = []
    for task_id, task_data in enumerate(task_partitions, start=1):
        if client_schedule[task_id - 1] > len(trainer.collaborators):
            trainer.expand_collaborators(client_schedule[task_id - 1])

        logs = trainer.train_task(
            task_id=task_id,
            task_data=task_data,
            retention_ratio=args.retention_ratio,
            verbose=True,
        )
        x_test, y_test = test_sets[task_id - 1]
        metrics = trainer.evaluate_all(x_test, y_test)
        record = {"task": task_id, **logs, **metrics}
        history.append(record)
        print(json.dumps(record, indent=2))

        if analyzer is not None:
            col0 = trainer.collaborators[0]
            if col0.x_val is not None and len(col0.x_val) > 0:
                explanations = analyzer.explain(
                    col0.model,
                    col0.x_val,
                    feature_names=feature_names,
                    class_names=class_names,
                    seed=args.seed + task_id,
                )
                frame = analyzer.to_frame(explanations)
                explain_path = output_dir / f"task_{task_id}_collaborator_0_explanations.csv"
                frame.to_csv(explain_path, index=False)
                print(f"Saved local explanations to {explain_path}")

    with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved metrics to {output_dir / 'metrics_history.json'}")


if __name__ == "__main__":
    main()
