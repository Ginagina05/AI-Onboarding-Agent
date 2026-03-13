# AI Onboarding Agent

An AI-powered codebase onboarding tool that scans a repository, extracts useful technical insights (business rules, APIs, database hints, configs, etc.), and generates an automated developer onboarding report.

It also provides:

Local semantic search over the codebase

AI-assisted chat with the repository

General AI chat for developer questions

The project consists of two main components:

onboarding_agent.py – Core analysis engine and CLI tool

app.py (Streamlit UI) – Web interface to interact with the agent

# Features
Codebase Indexing

Scans a repository and extracts structured snippets including:

## Business rules

Validation logic

API routes

SQL schema definitions

ORM models

Config variables

Imports / dependencies

Context chunks for semantic search

All extracted snippets are stored in a searchable index.

Automated Onboarding Report

Generates a Markdown onboarding document summarizing:

Business rules

API surface

Database schema hints

Configuration variables

Imports and dependencies

Example context snippets from the codebase

This report helps new developers quickly understand a project.

Local Codebase Q&A

The tool builds a TF-IDF vector index of the extracted snippets and allows developers to:

Ask questions about the codebase

Retrieve relevant code snippets

Explore the project faster

This works locally without needing AI APIs.

AI Assisted Chat

When an OpenAI API key is configured, the tool can:

Answer questions using GPT with codebase context

Provide general AI chat for developers

Architecture
                ┌────────────────────┐
                │   Streamlit UI     │
                │     app.py         │
                └─────────┬──────────┘
                          │
                          │ Calls CLI commands
                          ▼
                ┌────────────────────┐
                │ onboarding_agent.py│
                │  Core engine       │
                └─────────┬──────────┘
                          │
          ┌───────────────┼─────────────────┐
          ▼               ▼                 ▼
     Repo Scanner   Snippet Indexer   Report Generator
          │               │                 │
          ▼               ▼                 ▼
      index.jsonl     TF-IDF vectors    onboarding_report.md

Script Overview
1. onboarding_agent.py

This is the core engine responsible for scanning and analyzing the repository.

Responsibilities

Scan files in a repository

Extract meaningful snippets

Detect business rules and validations

Identify APIs and database definitions

Build a local search index

Generate onboarding reports

Provide CLI-based Q&A

Supported Languages

Python

JavaScript

TypeScript

Java

SQL

JSON

YAML

TOML

ENV files

Extracted Information

The scanner detects:

Type	Description
Business Rules	Validation logic, rule keywords, numeric thresholds
API Routes	Framework route decorators
Config	Environment variables and feature flags
Database	SQL DDL and ORM models
Imports	Dependencies and modules
Context	General code chunks for semantic search
CLI Commands
Index a Repository
python onboarding_agent.py index --root <repo_path> --out ./ai_onboarding_out


Example:

python onboarding_agent.py index --root ./my_project --out ./ai_onboarding_out


Output:

ai_onboarding_out/index.jsonl

Generate Onboarding Report
python onboarding_agent.py report --index ./ai_onboarding_out/index.jsonl --out ./ai_onboarding_out


Output:

ai_onboarding_out/onboarding_report.md

Local Codebase Chat

Ask questions about the indexed codebase.

python onboarding_agent.py chat --index ./ai_onboarding_out/index.jsonl


Example:

? how is user validation handled


The system returns relevant snippets.

GPT Chat With Repository Context

Requires OPENAI_API_KEY.

python onboarding_agent.py chatgpt --index ./ai_onboarding_out/index.jsonl

Ask General AI Questions
python onboarding_agent.py general "What is dependency injection?"

2. Streamlit Interface (app.py)

The Streamlit application provides a web interface for interacting with the onboarding agent.

Run it with:

streamlit run app.py

UI Features
Index Codebase

Allows users to input a repository path and run the indexing process.

Internally executes:

onboarding_agent.py index

Generate Report

Creates the onboarding report from the generated index.

Internally runs:

onboarding_agent.py report

View Onboarding Report

Displays the generated Markdown report directly in the UI.

Local Codebase Chat

Users can ask questions like:

How does authentication work?
Where are database models defined?
What APIs exist?


The system retrieves relevant snippets from the index.

General AI Chat

Provides a developer assistant powered by:

GPT-4.1-mini


Used for general programming or system questions.

Output Files

All generated files are stored in:

ai_onboarding_out/

File	Description
index.jsonl	Extracted snippets from the codebase
onboarding_report.md	Generated developer onboarding report
Installation
1. Install Dependencies
pip install streamlit openai httpx

2. Set OpenAI API Key (optional)
export OPENAI_API_KEY=your_api_key


Without the API key the system will still support:

indexing

report generation

local Q&A

Example Workflow
1. Start UI
streamlit run app.py

2. Enter repository path
/path/to/project

3. Click
Index Codebase

4. Click
Generate Report

5. Ask questions about the code
Where are pricing rules defined?

Use Cases

This tool is useful for:

Developer onboarding

Understanding legacy systems

Codebase exploration

Architecture discovery

Extracting hidden business logic

AI-assisted developer documentation

Future Improvements

Possible enhancements:

vector embeddings instead of TF-IDF

architecture diagrams

dependency graphs

microservice detection

automatic README generation

pull request analysis

GitHub integration