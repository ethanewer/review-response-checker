import asyncio
import base64
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Optional

from openai import AsyncClient
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
client = AsyncClient()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--responses", type=str, default="responses", help="Path to review responses.")
    parser.add_argument("--reviews", type=str, default="reviews", help="Path to reviews.")
    parser.add_argument("--paper", type=str, default="main.pdf", help="Path to paper.")
    return parser.parse_known_args()[0]


def load_inputs(path: str | Path) -> dict[str, str]:
    if isinstance(path, str):
        path = Path(path)

    assert path.is_dir()
    return {child: (path / child).read_text() for child in os.listdir(path)}


async def parse_review(review: str) -> list[str]:
    class ParseReviewOutput(BaseModel):
        comments: list[str]

    messages = [
        {
            "role": "system",
            "content": "You are a research assistant helping with scientific review. Your task is to create a comprehensive list of all questions, weaknesses, limitations, and all other criticism present in a scientific review.",
        },
        {
            "role": "user",
            "content": f"Extract all comments (questions, weaknesses, limitations, ect.) from the following review.\n\nREVIEW:\n\n{review}",
        },
    ]

    completion = await client.chat.completions.parse(
        model="gpt-4.1",
        messages=messages,  # type: ignore
        response_format=ParseReviewOutput,
    )

    assert completion.choices[0].message.parsed is not None
    return completion.choices[0].message.parsed.comments


async def check_response(comment: str, response: str, paper: Optional[bytes] = None) -> bool:
    class CheckResponseOutput(BaseModel):
        reasoning: str
        comment_is_fully_addressed: bool

    system_prompt = (
        "You are a research assistant helping with scientific review. "
        "Your task is to determine if a review response fully addresses a given/criticism."
        "You will be given the following information:"
        "\n- Comment within <comment></comment> tags"
        "\n- Response within <response></response> tags"
    )

    user_text = f"<comment>\n{comment}\n</comment>\n\n<response>\n{response}\n</response>\n\nDoes the response fully address the comment?"

    if paper is None:
        user_content = user_text
    else:
        user_content = [
            {
                "type": "text",
                "text": user_text,
            },
            {
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": f"data:application/pdf;base64,{base64.b64encode(paper).decode('utf-8')}",
                },
            },
        ]
        system_prompt += "\n- Full paper as pdf file"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content,
        },
    ]

    completion = await client.chat.completions.parse(
        model="o4-mini",
        messages=messages,  # type: ignore
        reasoning_effort="high",
        response_format=CheckResponseOutput,
    )

    assert completion.choices[0].message.parsed is not None
    return completion.choices[0].message.parsed.comment_is_fully_addressed


async def process_review(review: str, response: str, paper: Optional[bytes], n=16) -> dict[str, float]:
    comments = (await parse_review(review)) * n
    results = await asyncio.gather(*[check_response(comment, response, paper) for comment in comments])
    scores = defaultdict(int)
    for comment, result in zip(comments, results):
        scores[comment] += result

    return {comment: 100 * score / n for comment, score in scores.items()}


def print_scores(scores: dict[str, float]) -> None:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Comment", style="dim", width=60)
    table.add_column("Addressed (%)", justify="center")
    table.add_column("Score", justify="center")

    for comment, score in scores.items():
        icon = "✅" if score == 100 else ("⚠️" if score > 50 else "❌")
        table.add_row(comment, f"{score:.0f}% {icon}")

    panel = Panel(table, title="Response Coverage", expand=False, border_style="green")
    console.print(panel)


async def main() -> None:
    args = parse_args()
    reviews = load_inputs(args.reviews)
    responses = load_inputs(args.responses)
    paper = None if args.paper is None else Path(args.paper).read_bytes()

    assert reviews.keys() == responses.keys(), "Every review file must have a response file with the same name."
    keys = reviews.keys()

    results = await asyncio.gather(*[process_review(reviews[key], responses[key], paper) for key in keys])

    for key, scores in zip(keys, results):
        console.rule(f"[bold blue]{key.strip('.txt')}")
        print_scores(scores)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user.\n")
