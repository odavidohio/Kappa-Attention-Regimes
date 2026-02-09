# Contributing to Kappa-LLM

Thank you for your interest in contributing to **Kappa-LLM**!

Kappa-LLM is an **active research project**. Contributions are welcome, but please note that the public API, internal structure, and experimental scope are still evolving.

This document outlines how to contribute in a way that is constructive, reproducible, and aligned with the scientific goals of the project.

---

## 🤝 How to Contribute

We welcome contributions in the following areas:

### 🔧 Code Contributions
- Improvements to existing observables
- Performance optimizations
- Bug fixes
- Refactoring and cleanup
- Experimental integrations with additional models (research-grade)

### 📊 Experimental Contributions
- Additional benchmarks or datasets
- Cross-model or cross-lingual validation
- Ablation studies
- Proposals for new observables (e.g., Γ)

### 📖 Documentation
- Clarifications and corrections
- Usage examples
- Tutorials and explanatory notes
- Translation of documentation

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/odavidohio/Kappa-Attention-Regimes.git
cd Kappa-Attention-Regimes
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

> ⚠️ Development tooling such as linters, formatters, and pre-commit hooks  
> may be added in future releases. For now, keep changes minimal and focused.

---

## 💻 Development Guidelines

### Code Style

- Follow **PEP 8**
- Prefer clarity over cleverness
- Add docstrings to public functions and classes
- Avoid breaking existing interfaces without prior discussion

### Testing

Formal unit testing infrastructure is **under development**.

If you add new functionality:
- Include minimal validation or example scripts when possible
- Clearly document assumptions, limitations, and expected behavior

---

## 📝 Commit Guidelines

We follow a lightweight, Conventional Commits–inspired format:

```
<type>(<scope>): <short description>
```

**Types:**
- `feat` – new feature
- `fix` – bug fix
- `docs` – documentation
- `refactor` – code restructuring
- `test` – experimental or validation code
- `chore` – maintenance tasks

**Example:**
```
feat(observables): add experimental gradient flow proxy
```

---

## 🔬 Adding New Models or Experiments

Support for additional models or datasets is considered **experimental**.

Before implementing:
1. Open an issue describing the proposal
2. Explain:
   - Target model or dataset
   - Motivation
   - Expected outcomes
3. Keep implementations modular, isolated, and well-documented

---

## 🔄 Pull Requests

Before submitting a pull request:

- Ensure the code runs as expected
- Avoid breaking existing functionality
- Update relevant documentation
- Clearly describe:
  - What the PR does
  - Why it is needed
  - How it was validated

Maintainers may request revisions to ensure conceptual consistency and reproducibility.

---

## 🐛 Bug Reports

If you encounter a bug:
1. Check existing issues
2. Provide a minimal reproducible example
3. Include environment details (OS, Python version, model)

---

## 💡 Feature Requests

Feature requests are welcome, especially when motivated by:
- Empirical evidence
- Theoretical grounding
- Reproducibility or robustness concerns

Please open an issue with a clear and concise description.

---

## 📧 Questions and Discussion

- Use GitHub Discussions for open questions
- Or contact: **odavidohio@gmail.com**

---

## 🙏 Recognition

Significant contributors may be:
- Acknowledged in release notes
- Credited in associated publications, when appropriate

Thank you for helping improve Kappa-LLM.
