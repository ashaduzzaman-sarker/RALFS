# ============================================================================
# File: ralfs/cli.py
# ============================================================================
"""
RALFS CLI - Command-line interface for Retrieval-Augmented Long-Form Summarization.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ralfs.core.config import load_config
from ralfs.core.logging import setup_logging, get_logger
from ralfs.data.processor import run_preprocessing
from ralfs.data.indexer import IndexBuilder
from ralfs.retriever import create_retriever
from ralfs.generator import create_generator
from ralfs.training.trainer import train_model
from ralfs.evaluation.main import run_evaluation
from ralfs.evaluation.human import create_human_eval_template
from ralfs.utils.io import load_json, save_json, load_jsonl, save_jsonl

app = typer.Typer(
    name="ralfs",
    help="🚀 RALFS: Retrieval-Augmented Long-Form Summarization",
    add_completion=False,
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)

console = Console()
logger = None


def init_logger():
    global logger
    if logger is None:
        setup_logging()
        logger = get_logger("ralfs.cli")


def handle_error(error: Exception, message: str = "Error") -> None:
    console.print(f"[bold red]❌ {message}:[/bold red] {error}")
    raise typer.Exit(code=1)


@app.command()
def preprocess(
    dataset: str = typer.Option("arxiv", "--dataset", "-d", help="Dataset name: arxiv, pubmed, govreport"),
    split: str = typer.Option("train", "--split", "-s", help="Data split: train, validation, test"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="Max samples (for debugging)"),
    force_download: bool = typer.Option(False, "--force-download", is_flag=True, help="Force re-download"),
    force_rechunk: bool = typer.Option(False, "--force-rechunk", is_flag=True, help="Force re-chunk"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Custom config YAML"),
):
    """📦 Preprocess dataset: download and chunk documents."""
    init_logger()
    console.print(f"[bold green]Preprocessing {dataset} ({split})...[/bold green]")

    try:
        cfg = load_config(config) if config else load_config()
        cfg.data.dataset = dataset
        cfg.data.split = split
        if max_samples is not None:
            cfg.data.max_samples = max_samples

        output_path = run_preprocessing(cfg, force_download=force_download, force_rechunk=force_rechunk)
        console.print(f"[bold green]✅ Chunks saved to:[/bold green] {output_path}")
    except Exception as e:
        handle_error(e, "Preprocessing failed")


@app.command()
def build_index(
    dataset: str = typer.Option("arxiv", "--dataset", "-d", help="Dataset name"),
    split: str = typer.Option("train", "--split", "-s", help="Data split"),
    force: bool = typer.Option(False, "--force", is_flag=True, help="Force rebuild index"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
):
    """🔍 Build retrieval indexes (FAISS + BM25)."""
    init_logger()
    console.print(f"[bold blue]Building indexes for {dataset}...[/bold blue]")

    try:
        cfg = load_config(config) if config else load_config()
        cfg.data.dataset = dataset
        cfg.data.split = split

        builder = IndexBuilder(cfg)
        indexes = builder.build_all_indexes(force_rebuild=force)

        console.print("[bold green]✅ Indexes built successfully:[/bold green]")
        for index_type, path in indexes.items():
            console.print(f" - {index_type}: {path}")
    except Exception as e:
        handle_error(e, "Index building failed")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(10, "--k", "-k", help="Number of results"),
    dataset: str = typer.Option("arxiv", "--dataset", "-d", help="Dataset name"),
    retriever_type: str = typer.Option("hybrid", "--retriever-type", "-r", help="Retriever type: dense, sparse, hybrid"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
):
    """🔍 Search for relevant chunks."""
    init_logger()
    console.print(f"[bold blue]Searching for:[/bold blue] '{query}'")

    try:
        cfg = load_config(config) if config else load_config()
        cfg.data.dataset = dataset
        cfg.retriever.type = retriever_type

        ret = create_retriever(cfg)
        ret.load_index()
        results = ret.retrieve(query, k=k)

        console.print(f"\n[bold green]Found {len(results)} results:[/bold green]\n")
        for result in results:
            console.print(f"[bold cyan]Rank {result.rank}:[/bold cyan] Score: {result.score:.4f}")
            console.print(f" {result.text[:200]}...")
            console.print()
    except Exception as e:
        handle_error(e, "Search failed")


@app.command()
def train(
    config: Path = typer.Option("configs/train/default.yaml", "--config", "-c", help="Training config"),
    dataset: str = typer.Option("arxiv", "--dataset", "-d", help="Dataset name"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Checkpoint directory"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project", "-w", help="W&B project name"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
):
    """🔥 Train RALFS model with LoRA and adaptive FiD."""
    init_logger()
    console.print("[bold red]Starting training...[/bold red]")

    try:
        cfg = load_config(config)
        cfg.data.dataset = dataset

        if output_dir:
            cfg.train.training.output_dir = str(output_dir) if hasattr(cfg.train, 'training') else str(output_dir)

        if wandb_project:
            if not hasattr(cfg.train, 'wandb'):
                from dataclasses import dataclass
                @dataclass
                class WandbConfig:
                    enabled: bool = True
                    project: str = "ralfs"
                cfg.train.wandb = WandbConfig()
            cfg.train.wandb.enabled = True
            cfg.train.wandb.project = wandb_project

        stats = train_model(cfg)

        console.print("[bold green]✅ Training complete![/bold green]")
        console.print(f"Final train loss: {stats['train_losses'][-1]:.4f}")
        if stats.get('eval_losses'):
            console.print(f"Final eval loss: {stats['eval_losses'][-1]:.4f}")
    except Exception as e:
        handle_error(e, "Training failed")


@app.command()
def generate(
    input_file: Path = typer.Argument(..., help="Input JSONL file with documents", exists=True, readable=True),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-m", help="Model checkpoint path", exists=True),
    output_file: Path = typer.Option("results/summaries.json", "--output", "-o", help="Output file"),
    dataset: str = typer.Option("arxiv", "--dataset", "-d", help="Dataset name (for retriever)"),
    retriever_k: int = typer.Option(20, "--retriever-k", "-k", help="Number of chunks to retrieve"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
):
    """✨ Generate summaries for input documents."""
    init_logger()
    console.print(f"[bold magenta]Generating summaries from {input_file}...[/bold magenta]")

    try:
        cfg = load_config(config) if config else load_config()
        cfg.data.dataset = dataset

        documents = load_jsonl(input_file) if input_file.suffix == '.jsonl' else load_json(input_file)
        console.print(f"Loaded {len(documents)} documents")

        retriever = create_retriever(cfg)
        retriever.load_index()

        generator = create_generator(cfg)
        generator.load_checkpoint(str(checkpoint))

        results = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Generating summaries...", total=len(documents))

            for doc in documents:
                doc_id = doc.get('id', 'unknown')
                text = doc.get('text', '')

                retrieved = retriever.retrieve(text[:1000], k=retriever_k)
                passages = [{'text': r.text, 'score': r.score} for r in retrieved]

                result = generator.generate(text[:1000], passages)

                results.append({
                    'id': doc_id,
                    'summary': result.summary,
                    'k_used': result.k_used,
                    'metadata': result.metadata,
                })
                progress.update(task, advance=1)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, output_file)
        console.print(f"[bold green]✅ Summaries saved to:[/bold green] {output_file}")
    except Exception as e:
        handle_error(e, "Generation failed")


@app.command()
def evaluate(
    predictions: Path = typer.Argument(..., help="Predictions JSON/JSONL", exists=True),
    references: Path = typer.Argument(..., help="References JSON/JSONL", exists=True),
    output_dir: Path = typer.Option("results/evaluation", "--output-dir", "-o", help="Output directory"),
    metrics: str = typer.Option("rouge,bertscore,egf", "--metrics", "-m", help="Comma-separated metrics"),
):
    """📊 Evaluate summaries with ROUGE, BERTScore, and EGF."""
    init_logger()
    console.print("[bold cyan]Evaluating summaries...[/bold cyan]")

    try:
        metrics_list = [m.strip() for m in metrics.split(',')]
        results = run_evaluation(
            predictions_path=predictions,
            references_path=references,
            output_dir=output_dir,
            metrics=metrics_list,
        )
        console.print("[bold green]✅ Evaluation complete![/bold green]")
        console.print(f"\nResults saved to: {output_dir}")
    except Exception as e:
        handle_error(e, "Evaluation failed")


@app.command()
def human_eval(
    predictions: Path = typer.Argument(..., help="Predictions JSON/JSONL", exists=True),
    references: Path = typer.Argument(..., help="References JSON/JSONL", exists=True),
    output_file: Path = typer.Option("results/human_eval.csv", "--output", "-o", help="Output CSV"),
    num_samples: int = typer.Option(50, "--num-samples", "-n", help="Number of samples"),
    randomize: bool = typer.Option(True, "--randomize", is_flag=True, help="Randomize sample selection"),
):
    """👥 Create human evaluation template (CSV)."""
    init_logger()
    console.print("[bold yellow]Creating human evaluation template...[/bold yellow]")

    try:
        pred_data = load_jsonl(predictions) if predictions.suffix == '.jsonl' else load_json(predictions)
        ref_data = load_jsonl(references) if references.suffix == '.jsonl' else load_json(references)

        output_path = create_human_eval_template(
            predictions=pred_data,
            references=ref_data,
            output_path=output_file,
            num_samples=num_samples,
            randomize=randomize,
        )
        console.print(f"[bold green]✅ Template saved to:[/bold green] {output_path}")
    except Exception as e:
        handle_error(e, "Human eval template creation failed")


@app.command()
def info(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file to display"),
):
    """ℹ️ Display RALFS system information."""
    init_logger()

    table = Table(title="RALFS System Info")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Version", "1.0.0")
    table.add_row("Python", "3.10+")
    table.add_row("PyTorch", "2.4.0+")
    table.add_row("Transformers", "4.44.2")

    console.print(table)

    if config:
        console.print("\n[bold]Configuration:[/bold]")
        cfg = load_config(config)
        config_table = Table(show_header=False)
        config_table.add_column("Key", style="cyan")
        config_table.add_column("Value", style="white")

        config_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__
        for key, value in config_dict.items():
            val_str = str(value)
            config_table.add_row(str(key), val_str[:100] + ("..." if len(val_str) > 100 else ""))

        console.print(config_table)


@app.command()
def pipeline(
    dataset: str = typer.Option("arxiv", "--dataset", "-d", help="Dataset name"),
    split: str = typer.Option("train", "--split", "-s", help="Data split for training"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="Max samples (for testing)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    skip_train: bool = typer.Option(False, "--skip-train", help="Skip training step"),
    skip_generate: bool = typer.Option(False, "--skip-generate", help="Skip generation step"),
    skip_evaluate: bool = typer.Option(False, "--skip-evaluate", help="Skip evaluation step"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-m", help="Model checkpoint (for generation)"),
):
    """🔄 Run complete pipeline: preprocess → index → train → generate → evaluate."""
    init_logger()
    console.print(f"[bold magenta]Running complete RALFS pipeline on {dataset} (max {max_samples or 'all'} samples)...[/bold magenta]")

    try:
        cfg = load_config(config) if config else load_config()
        cfg.data.dataset = dataset
        cfg.data.split = split
        if max_samples is not None:
            cfg.data.max_samples = max_samples

        # Step 1: Data Preprocessing
        console.print("\n[bold blue]Step 1: Data Preprocessing...[/bold blue]")
        preprocessed_path = run_preprocessing(cfg)
        console.print(f"[green]✓ Preprocessed data saved to: {preprocessed_path}[/green]")

        # Step 2: Index Building (for Retrieval)
        console.print("\n[bold blue]Step 2: Index Building (Retrieval)...[/bold blue]")
        builder = IndexBuilder(cfg)
        indexes = builder.build_all_indexes()
        console.print("[green]✓ Indexes built successfully[/green]")

        # Step 3: Training
        if not skip_train:
            console.print("\n[bold blue]Step 3: Training Model...[/bold blue]")
            train_stats = train_model(cfg)
            console.print(f"[green]✓ Training complete! Final loss: {train_stats['train_losses'][-1]:.4f}[/green]")
            # Use the latest checkpoint for generation if not specified
            if checkpoint is None:
                checkpoint = Path(cfg.train.training.output_dir if hasattr(cfg.train, 'training') else "checkpoints") / "best_model"
        else:
            console.print("\n[yellow]⊘ Step 3: Training skipped[/yellow]")

        # Step 4: Generation
        if not skip_generate and checkpoint and checkpoint.exists():
            console.print("\n[bold blue]Step 4: Generation...[/bold blue]")
            # Use validation split for generation
            cfg.data.split = "validation"
            val_data_path = Path("data/processed") / dataset / "validation" / "chunks.jsonl"
            
            if val_data_path.exists():
                retriever = create_retriever(cfg)
                retriever.load_index()
                generator = create_generator(cfg)
                generator.load_checkpoint(str(checkpoint))
                
                val_docs = load_jsonl(val_data_path)[:min(len(val_docs), max_samples or 100)]
                
                predictions_path = Path("results") / "predictions.json"
                predictions_path.parent.mkdir(parents=True, exist_ok=True)
                
                predictions = []
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                    task = progress.add_task("Generating summaries...", total=len(val_docs))
                    for doc in val_docs:
                        query = doc.get('text', '')[:1000]
                        retrieved = retriever.retrieve(query, k=20)
                        passages = [{'text': r.text, 'score': r.score} for r in retrieved]
                        result = generator.generate(query, passages)
                        predictions.append({
                            'id': doc.get('id', 'unknown'),
                            'summary': result.summary,
                            'k_used': result.k_used,
                        })
                        progress.update(task, advance=1)
                
                save_json(predictions, predictions_path)
                console.print(f"[green]✓ Generated {len(predictions)} summaries: {predictions_path}[/green]")
            else:
                console.print(f"[yellow]⊘ Validation data not found at {val_data_path}, skipping generation[/yellow]")
        elif skip_generate:
            console.print("\n[yellow]⊘ Step 4: Generation skipped[/yellow]")
        else:
            console.print("\n[yellow]⊘ Step 4: Generation skipped (no checkpoint available)[/yellow]")

        # Step 5: Evaluation
        if not skip_evaluate:
            predictions_path = Path("results") / "predictions.json"
            references_path = Path("data/processed") / dataset / "validation" / "references.json"
            
            if predictions_path.exists() and references_path.exists():
                console.print("\n[bold blue]Step 5: Evaluation...[/bold blue]")
                eval_results = run_evaluation(
                    predictions_path=predictions_path,
                    references_path=references_path,
                    output_dir=Path("results/evaluation"),
                    metrics=["rouge", "bertscore", "egf"]
                )
                console.print("[green]✓ Evaluation complete! Results saved to results/evaluation[/green]")
            else:
                console.print("\n[yellow]⊘ Step 5: Evaluation skipped (missing predictions or references)[/yellow]")
        else:
            console.print("\n[yellow]⊘ Step 5: Evaluation skipped[/yellow]")

        console.print("\n[bold green]✅ Pipeline complete![/bold green]")
        console.print("[cyan]Complete workflow executed: Data preprocessing → Index building → Retrieval → Generation → Evaluation[/cyan]")
    except Exception as e:
        handle_error(e, "Pipeline failed")


if __name__ == "__main__":
    app()
