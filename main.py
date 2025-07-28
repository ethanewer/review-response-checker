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
from rich.text import Text

console = Console()
client = AsyncClient()


class Typo(BaseModel):
    text: str
    description: str


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--reviews", type=str, help="Path to reviews.")
    parser.add_argument("--responses", type=str, help="Path to review responses.")
    parser.add_argument("--paper", type=str, default=None, help="Path to paper.")
    parser.add_argument("--n", type=int, default=8, help="Number of trials for judge LLM.")
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

    for i in range(4):
        try:
            completion = await client.chat.completions.parse(
                model="gpt-4.1",
                messages=messages,  # type: ignore
                response_format=ParseReviewOutput,
            )
            assert completion.choices[0].message.parsed is not None
            return completion.choices[0].message.parsed.comments
        except Exception:
            await asyncio.sleep(2**i)

    raise Exception("`parse_review` max retries exhausted.")


async def find_typos(review: str) -> list[Typo]:
    class ParseReviewOutput(BaseModel):
        typos: list[Typo]

    messages = [
        {
            "role": "system",
            "content": "You are a research assistant helping with scientific review. Your task is to check the following review response for any spelling, grammar, or other syntax mistakes.",
        },
        {
            "role": "user",
            "content": f"Find any typos in the following review.\n\nREVIEW:\n\n{review}",
        },
    ]

    for i in range(4):
        try:
            completion = await client.chat.completions.parse(
                model="gpt-4.1",
                messages=messages,  # type: ignore
                response_format=ParseReviewOutput,
            )
            assert completion.choices[0].message.parsed is not None
            return completion.choices[0].message.parsed.typos
        except Exception:
            await asyncio.sleep(2**i)

    raise Exception("`parse_review` max retries exhausted.")


async def check_response_to_comment(comment: str, response: str, paper: Optional[bytes] = None) -> bool:
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

    for i in range(4):
        try:
            completion = await client.chat.completions.parse(
                model="gpt-4.1",
                messages=messages,  # type: ignore
                response_format=CheckResponseOutput,
            )
            assert completion.choices[0].message.parsed is not None
            return completion.choices[0].message.parsed.comment_is_fully_addressed
        except Exception:
            await asyncio.sleep(2**i)

    raise Exception("`check_response_to_comment` max retries exhausted.")


async def process_review(review: str, response: str, paper: Optional[bytes], n=4) -> dict[str, float]:
    comments = (await parse_review(review)) * n
    results = await asyncio.gather(*[check_response_to_comment(comment, response, paper) for comment in comments])
    scores = defaultdict(int)
    for comment, result in zip(comments, results):
        scores[comment] += result

    return {comment: 100 * score / n for comment, score in scores.items()}


def print_results(name: str, response: str, typos: list[Typo], scores: dict[str, float]) -> None:
    from rich.console import Group

    table = Table(show_header=True, show_lines=True, header_style="bold cyan")
    table.add_column("Review Comment", style="dim", width=90)
    table.add_column("Score (%)", justify="center", width=10)

    for comment, score in scores.items():
        icon = "✅" if score >= 80 else ("⚠️" if score >= 50 else "❌")
        table.add_row(comment.strip(), f"{score:.0f}% {icon}")

    if typos:
        typos_table = Table(show_header=True, show_lines=True, header_style="bold magenta")
        typos_table.add_column("Typo", style="yellow", width=50)
        typos_table.add_column("Description", style="white", width=50)
        for typo in typos:
            typos_table.add_row(typo.text, typo.description)
    else:
        typos_table = Text.from_markup("[bold green]No typos found.")

    if len(response) > 10000:
        len_message = f"[bold red]{len(response)}/10000 characters used."
    else:
        len_message = f"[bold green]{len(response)}/10000 characters used."

    len_message_text = Text.from_markup(len_message)

    group = Group(table, typos_table, len_message_text)
    panel = Panel(group, title=name, expand=False, border_style="green")
    console.print(panel)


async def main() -> None:
    args = parse_args()
    reviews = load_inputs(args.reviews)
    responses = load_inputs(args.responses)
    paper = None if args.paper is None else Path(args.paper).read_bytes()

    assert reviews.keys() == responses.keys(), "Every review file must have a response file with the same name."
    keys = reviews.keys()

    all_typos = await asyncio.gather(*[find_typos(responses[key]) for key in keys])
    all_scores = await asyncio.gather(*[process_review(reviews[key], responses[key], paper, args.n) for key in keys])

    for key, typos, scores in zip(keys, all_typos, all_scores):
        print_results(key.strip(".txt"), responses[key], typos, scores)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user.\n")
