# Review Response Checker

A CLI tool to validate that every review file has a corresponding response file.

---

## 📦 Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/ethanewer/review-response-checker.git
   cd review-response-checker
   uv venv --python=3.11 .venv
   uv pip install -e .
   ```
---

## 🔑 Configuration

This tool uses the OpenAI Python client. You must set up your OpenAI API credentials as environment variables before running any commands.

1. Create an API key on the [OpenAI dashboard](https://platform.openai.com/).
2. In your shell, export the following variables:

   ```sh
   export OPENAI_API_KEY="your_api_key_here"
   ```

## 📁 Directory Structure

* `reviews/`: Directory containing one review file per submission.
* `responses/`: Directory containing one response file per review.

> **Note:** Every file in `reviews/` **must** have a matching file (same name) in `responses/`.

## 🚀 Usage

### Basic command
```sh
uv run main.py --reviews path/to/reviews/dir \
               --responses path/to/responses/dir
```

### With paper pdf and specific number of trials with judge LLM

```sh
uv run main.py --reviews path/to/reviews/dir \
               --responses path/to/responses/dir \
               --paper path/to/paper.pdf \
               --n 32
```


