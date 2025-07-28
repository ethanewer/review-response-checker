# Review Response Checker

```sh
uv run main.py --reviews path/to/reviews/dir --responses path/to/responses/dir

uv run main.py --reviews path/to/reviews/dir --responses path/to/responses/dir --paper path/to/paper.pdf --n 32
```

- Reviews should be in a directory with one review per file
- Responses should be in a directory with one response per file
- Every review file must have a response file with the same name