# Code Quality and Linting Setup

This project uses modern Python tooling to maintain high code quality and catch issues early.

## 🔧 Tools Used

### **Ruff** - Lightning Fast Python Linter & Formatter
- **Speed**: 10-100x faster than traditional tools (flake8, black, isort)
- **Comprehensive**: Combines functionality of multiple tools in one
- **Features**:
  - Detects unused imports and variables ✨
  - Enforces Python conventions (PEP 8)
  - Auto-fixes many issues
  - Import sorting and formatting
  - Security vulnerability detection

### **Bandit** - Security Linting
- Scans for common security issues
- Detects hardcoded passwords, SQL injection risks, etc.

### **Pre-commit** - Automated Quality Gates
- Runs linting automatically before commits
- Prevents bad code from entering the repository
- Fast feedback loop for developers

## 🚀 Quick Start

```bash
# 1. Install development dependencies
uv sync --extra dev

# 2. Install pre-commit hooks (one-time setup)
uv run pre-commit install

# 3. Check your code
uv run ruff check

# 4. Auto-fix issues
uv run ruff check --fix

# 5. Format code
uv run ruff format
```

## 📋 Available Commands

### Manual Linting
```bash
# Check for issues (no changes made)
uv run ruff check

# Check specific files
uv run ruff check llama3.py tokenizer.py

# Auto-fix issues where possible
uv run ruff check --fix

# Format code according to style guide
uv run ruff format

# Security scan
uv run bandit -r .
```

### Convenience Script
Use the provided script for common workflows:

```bash
# Run all checks and fixes
python scripts/lint.py --all

# Just check (no fixes)
python scripts/lint.py --check

# Just auto-fix
python scripts/lint.py --fix

# Security check only
python scripts/lint.py --security

# Check specific files
python scripts/lint.py --check llama3.py model_config.py
```

### Pre-commit Integration
```bash
# Install hooks (one-time setup)
uv run pre-commit install

# Run hooks on all files manually
uv run pre-commit run --all-files

# Hooks run automatically on git commit
git commit -m "Your changes"  # Linting runs automatically
```

## ⚙️ Configuration

### Ruff Configuration (`pyproject.toml`)
- **Line length**: 120 characters
- **Target Python**: 3.12+
- **Enabled rules**:
  - Pyflakes (unused imports/variables)
  - PEP 8 style checks
  - Import sorting
  - Complexity analysis
  - Security checks
  - Code simplification suggestions

### What Gets Checked
✅ **Unused imports and variables**
✅ **PEP 8 style violations**
✅ **Import sorting and organization**
✅ **Security vulnerabilities**
✅ **Code complexity**
✅ **Potential bugs (like comparing with `True/False`)**
✅ **Performance improvements**

### What Gets Ignored
❌ Docstring requirements (ML code focused)
❌ Print statements (useful for ML debugging)
❌ Complex functions (ML algorithms can be complex)
❌ Magic numbers (common in ML)

## 🔄 Development Workflow

1. **Write code** as usual
2. **Run linter**: `python scripts/lint.py --fix`
3. **Commit**: Pre-commit hooks run automatically
4. **Push**: Clean, high-quality code

## 📊 Benefits

- **Catch bugs early**: Unused variables, incorrect comparisons, etc.
- **Consistent style**: Automatic formatting and import organization
- **Security**: Detect potential vulnerabilities
- **Performance**: Suggestions for faster code patterns
- **Maintainability**: Cleaner, more readable code
- **Team productivity**: Fewer code review discussions about style

## 🏃‍♂️ Performance

Ruff is extremely fast:
- **Lints entire codebase in <100ms**
- **No waiting**: Instant feedback
- **Pre-commit hooks are fast**: Won't slow down your workflow

## 🛠️ Troubleshooting

### Common Issues

**Q: Ruff complains about line length**
A: Configure your editor to show the 120-character limit, or use `ruff format` to auto-wrap

**Q: Pre-commit hook fails**
A: Run `python scripts/lint.py --fix` to auto-fix issues, then commit again

**Q: Want to ignore a specific warning?**
A: Add `# noqa: RULE_CODE` to the line, or update `pyproject.toml` configuration

**Q: Need to skip pre-commit temporarily?**
A: Use `git commit --no-verify` (not recommended for regular use)

## 📚 Learn More

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python PEP 8 Style Guide](https://peps.python.org/pep-0008/)