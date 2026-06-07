# nophigene

Local-first DRD4 analysis workbench with an optional Docker path.

The project now has two supported run modes:

- local mode: the default and recommended workflow, launched from the repaired `.venv`
- Docker mode: a slimmer secondary option for reproducible container runs

## Why the workflow changed

This app is currently a single-process Python web app plus a CLI pipeline:

- [src/webapp.py](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/src/webapp.py:1) provides the Flask UI
- [src/analysis.py](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/src/analysis.py:1) contains the reusable analysis workflow
- [src/app.py](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/src/app.py:1) dispatches to either web or CLI mode

Because the current app does not need multiple services, a database, or orchestration, Docker was adding more local overhead than value. The biggest pain points were:

- large build context
- long image build times
- heavy scientific dependencies getting pulled into every app build
- Docker Desktop startup latency for simple local runs

The repo is now structured so:

- local launch is the default
- Docker is still available, but slimmer
- app runtime dependencies are separated from optional research extras

## Dependency layout

The dependency files are now split by purpose:

- [requirements-app.txt](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/requirements-app.txt:1)
  - minimal runtime set for the UI, CLI, and tests
- [requirements-research.txt](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/requirements-research.txt:1)
  - optional heavier packages for exploratory or future workflows
- [requirements.txt](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/requirements.txt:1)
  - convenience alias to the app requirements

At the moment, the app runtime uses:

- `Flask`
- `pandas`
- `numpy`
- `scikit-allel`
- `methylprep`
- `requests`

Moved out of the default app runtime:

- `deepchem`
- `biomart`
- `matplotlib`
- `pysam`

Important note:

- `pysam` remains listed as an optional research dependency, but it still does not install cleanly on this Windows setup

## Launchers

### Default local launchers

These are the main “starter icon” files for day-to-day use on Windows:

- [Start NophiGene UI.cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Start%20NophiGene%20UI.cmd)
- [Stop NophiGene UI.cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Stop%20NophiGene%20UI.cmd)

They call:

- [scripts/start_nophigene_ui_local.ps1](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/scripts/start_nophigene_ui_local.ps1:1)
- [scripts/stop_nophigene_ui_local.ps1](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/scripts/stop_nophigene_ui_local.ps1:1)

What the local start launcher does:

- checks that `.venv\Scripts\python.exe` exists
- checks that key app dependencies can be imported
- creates `data/`, `data/reference/hg38/`, `data/extracted/`, and `results/` if needed
- starts the UI from the local environment
- selects an available local port starting at `8766`
- waits for the server health check to respond on the selected port
- opens the browser automatically
- tracks the running process in a local PID file
- keeps BAM extraction disabled unless you explicitly start the PowerShell launcher with `-EnableLocalExtraction` and local `samtools`/`bcftools` are on PATH
- supports the Extraction tab's native **Browse BAM File** picker when running locally

What the local stop launcher does:

- stops the tracked local UI process
- removes the PID file
- leaves reference files and extracted VCFs in `data/`
- exits cleanly if nothing is running

### Secondary Docker launchers

These are still available if you want a containerized run:

- [Start NophiGene UI (Docker).cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Start%20NophiGene%20UI%20%28Docker%29.cmd)
- [Stop NophiGene UI (Docker).cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Stop%20NophiGene%20UI%20%28Docker%29.cmd)

They call:

- [scripts/start_nophigene_ui.ps1](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/scripts/start_nophigene_ui.ps1:1)
- [scripts/stop_nophigene_ui.ps1](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/scripts/stop_nophigene_ui.ps1:1)

## Local setup

### Recommended Python version

Use Python `3.10`.

The current project `.venv` has already been repaired to point to Python `3.10.11`.

### Install app dependencies

If you need to recreate the local environment from scratch:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

### Install optional research extras

Only do this if you need the non-runtime stack:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

## Local-first usage

### Fastest path

Double-click:

- [Start NophiGene UI.cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Start%20NophiGene%20UI.cmd)

Then open:

- [http://127.0.0.1:8766](http://127.0.0.1:8766)

`8766` is the preferred port. If it is occupied, the launcher selects the next bindable port and prints the actual URL. An explicitly requested port, such as `-Port 9000`, fails immediately when unavailable instead of waiting for the startup timeout. The selected local port is recorded in `.nophigene-ui.port`.

When finished, double-click:

- [Stop NophiGene UI.cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Stop%20NophiGene%20UI.cmd)

### Manual local launch

You can also run the UI directly:

```powershell
.\.venv\Scripts\python.exe src\app.py web --host 127.0.0.1 --port 8766
```

Or run the CLI path:

```powershell
.\.venv\Scripts\python.exe src\app.py cli --vcf data/drd4.vcf.gz --idat data/202277800037_R01C01 --out results/drd4_report.html
```

## Expected input layout

The UI and CLI both assume a local project structure like this:

```text
data/
  drd4.vcf.gz
  202277800037_R01C01_Grn.idat
  202277800037_R01C01_Red.idat
results/
```

Important:

- the IDAT argument or form field uses the shared prefix only
- example: `data/202277800037_R01C01`

## GRCh38 BAM extraction

The UI includes an **Extraction** tab for BAM-to-VCF prep before running analysis. This path is Docker-only by default because it requires command-line genomics tools:

- `samtools`
- `bcftools`

The Extraction tab can:

- prepare the UCSC hg38 analysis-set reference under `data/reference/hg38/`
- download `hg38.analysisSet.fa.gz`
- verify it against UCSC `md5sum.txt`
- decompress it to `hg38.analysisSet.fa`
- create `hg38.analysisSet.fa.fai`
- search a selected folder tree for `.bam` files and add matching paths to the BAM picker
- open a native **Browse BAM File** picker in local mode; Docker mode cannot open host file windows from inside the container
- call a regional VCF from a GRCh38/hg38 BAM into `data/extracted/`
- populate the Run Analysis VCF field with the extracted file

Inputs expected for extraction:

- a coordinate-sorted BAM under `data/`
- a BAM index, or permission for the app to create one with `samtools index`
- a BAM aligned to GRCh38/hg38, not hg19

The extractor resolves contig naming automatically for common aliases such as `15` versus `chr15`, and `MT` versus `chrM`.

When preprocessing resolves a gene whose bundled knowledge base is hg38-only, the UI shows a GRCh38 extraction suggestion and pre-fills the Extraction tab for that gene.

For BAM extraction, prefer the Docker launcher so the required tools are present:

```powershell
.\Start NophiGene UI (Docker).cmd
```

You can also start the same Docker/samtools/bcftools runtime through the local starter by passing `-UseDocker`:

```powershell
.\Start NophiGene UI.cmd -UseDocker
```

If you have `samtools` and `bcftools` installed locally and want to opt into local extraction explicitly:

```powershell
.\Start NophiGene UI.cmd -EnableLocalExtraction
```

## What the UI writes

Each run creates:

- a report file at the path you choose
- a companion methylation CSV beside the report

Example:

- requested report: `results/drd4_report.html`
- generated methylation file: `results/drd4_report_methylation.csv`

## Local REST API

The Flask process also serves a versioned local API:

- [http://127.0.0.1:8766/api/v1](http://127.0.0.1:8766/api/v1)
- OpenAPI: [http://127.0.0.1:8766/api/v1/openapi.json](http://127.0.0.1:8766/api/v1/openapi.json)
- Health: [http://127.0.0.1:8766/api/v1/health](http://127.0.0.1:8766/api/v1/health)

Version 1 is intended for trusted local use. It references local filesystem paths and does not provide authentication or uploads.

### Create a sample profile

Profiles persist under `data/api/sample_profiles.json`. They describe one reusable IDAT pair, one full methylation manifest, and assembly-labelled VCF or BAM sources.

```powershell
curl.exe -X POST http://127.0.0.1:8766/api/v1/profiles `
  -H "Content-Type: application/json" `
  --data-binary '@profile.json'
```

Example `profile.json`:

```json
{
  "id": "sample-202277800037",
  "display_name": "Sample 202277800037",
  "default_genome_build": "hg19",
  "idat_prefix": "data/202277800037_R01C01",
  "manifest_path": "data/infinium-methylationepic-v-1-0-b5-manifest-file.csv",
  "population_statistics_path": "",
  "vcf_sources": [
    {
      "path": "data/GFXC926398.filtered.snp.vcf.gz",
      "genome_build": "hg19"
    }
  ],
  "bam_sources": [
    {
      "path": "data/sample_hg38.bam",
      "genome_build": "hg38"
    }
  ]
}
```

For a non-hg38 BAM source, include a matching `reference_fasta` in that source object. The built-in hg38 extraction path uses `data/reference/hg38/hg38.analysisSet.fa`.

### Submit a full workflow

Jobs accept one gene or up to 100 unique genes. Symbols are normalized to uppercase and duplicate names are removed.

```powershell
curl.exe -X POST http://127.0.0.1:8766/api/v1/jobs `
  -H "Content-Type: application/json" `
  -d '{
    "operation": "full_workflow",
    "profile_id": "sample-202277800037",
    "genes": ["DRD4", "HERC2", "POTEB3"],
    "analysis_scope": "promoter_plus_gene",
    "genome_build": "auto",
    "options": {
      "update_general_database": false,
      "overwrite_general_database": false
    }
  }'
```

The response is `202 Accepted` with a job ID. Poll it and fetch the final result:

```powershell
curl.exe http://127.0.0.1:8766/api/v1/jobs/JOB_ID
curl.exe http://127.0.0.1:8766/api/v1/jobs/JOB_ID/result
curl.exe -o artifacts.zip http://127.0.0.1:8766/api/v1/jobs/JOB_ID/artifacts/artifacts.zip
```

Each successful gene in a full workflow produces:

```text
results/api/jobs/JOB_ID/genes/GENE/
  report.html
  report.json
  report_summary.csv
  variants.csv
  methylation.csv
  analysis.json
  manifest.csv
  region.json
```

`report.json` is the canonical machine-readable report. It includes its schema version, region and source provenance, interpreted variants, methylation insights, population context, predictive theses, warnings, and artifact links.

### Independent operations

The supported `operation` values are:

- `resolve_regions`
- `prepare_manifests`
- `extract_variants`
- `analyze`
- `render_reports`
- `full_workflow`

`analyze` may reuse regions and extracted VCFs from a completed `source_job_id`. `render_reports` requires a completed analysis source job. A batch continues when one gene fails and finishes with `partial` status when it contains both successes and failures.

Only queued jobs can be cancelled:

```powershell
curl.exe -X POST http://127.0.0.1:8766/api/v1/jobs/JOB_ID/cancel
```

Queued jobs and completed results survive app restarts. A job interrupted while running is marked failed with the `interrupted` error code.

## Genotype-aware variant interpretation

The variant layer decodes sample-level VCF `GT` before making any phenotype-oriented statement. `REF` and `ALT` are kept as the site definition, while the sample genotype is reported separately as decoded alleles, zygosity, ALT dosage, call-quality flags, and confidence.

See [docs/genotype-aware-variant-analysis.md](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/docs/genotype-aware-variant-analysis.md:1) for the design note explaining why `GT` is authoritative, why `INFO/AF`, `INFO/AC`, and `INFO/AN` are not substitutes for sample genotype, and why phenotype predictions use dosage plus QC uncertainty.

## Docker is now secondary

Docker still works, but it is no longer the recommended local default.

### What changed to make Docker lighter

- the image now installs from [requirements-app.txt](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/requirements-app.txt:1) instead of the full research stack
- the runtime image installs `samtools` and `bcftools` for the Docker-only Extraction tab
- [Dockerfile](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Dockerfile:1) now copies only `src/` and the app requirements into the image
- [.dockerignore](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/.dockerignore:1) now excludes:
  - `data/`
  - `results/`
  - `.venv/`
  - `.docker-local/`
  - `.pytest_cache/`
  - notebooks, tests, and launcher scripts

That should materially reduce Docker build context size and image churn.

### Build the Docker image manually

```bash
docker build -t nophigene:latest .
```

### Run Docker manually

```bash
docker run --rm -it \
  -p 8766:8766 \
  -e NOPHIGENE_IN_DOCKER=1 \
  -v "${PWD}/data":/home/appuser/app/data \
  -v "${PWD}/results":/home/appuser/app/results \
  nophigene:latest
```

Then open:

- [http://127.0.0.1:8766](http://127.0.0.1:8766)

### Use the Docker launcher

If you still want the automated Docker flow, double-click:

- [Start NophiGene UI (Docker).cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Start%20NophiGene%20UI%20%28Docker%29.cmd)

When done, use:

- [Stop NophiGene UI (Docker).cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Stop%20NophiGene%20UI%20%28Docker%29.cmd)

## VS Code

VS Code is now aligned with the local-first setup:

- [settings.json](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/.vscode/settings.json:1) points to `.venv\Scripts\python.exe`
- [launch.json](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/.vscode/launch.json:1) includes:
  - a local UI launch config
  - a CLI launch config

## Troubleshooting

### Double-clicking the local launcher says dependencies are missing

Reinstall the app runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

### The browser does not open automatically

Open:

- [http://127.0.0.1:8766](http://127.0.0.1:8766)

### The local server failed to start

Check the local launcher logs in the repo root:

- `.nophigene-ui.log`
- `.nophigene-ui.err.log`

### Docker is still slow

That is now expected to be less severe than before, but local `.venv` launch is still the recommended path for iterative work.

### `pysam` still fails on Windows

That package remains optional and is not required for the current UI or CLI path.

## Recommended workflow now

For daily use:

1. Put your input files in `data/`.
2. Double-click [Start NophiGene UI.cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Start%20NophiGene%20UI.cmd).
3. Run the analysis from the browser.
4. Open outputs from `results/`.
5. Double-click [Stop NophiGene UI.cmd](/C:/Users/Mewxy/Desktop/YouTopy/NophiGene/nophigene-drd4-analysis/Stop%20NophiGene%20UI.cmd) when finished.
