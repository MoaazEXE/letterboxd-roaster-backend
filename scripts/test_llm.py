import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm import init_roast_client


async def main() -> None:
    client = init_roast_client()
    examples = [
        ("Interstellar", "Arrival", 4.7),
        ("The Notebook", "A Walk to Remember", 3.9),
    ]
    for inp, rec, rating in examples:
        roast = await client.generate_roast(inp, rec, rating)
        print(f"INPUT: {inp} -> RECOMMEND: {rec} ({rating})\nROAST: {roast}\n{'-' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
