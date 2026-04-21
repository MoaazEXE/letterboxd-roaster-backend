import sys
import pathlib
# ensure project root is on sys.path when running the script directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
from backend import llm

async def main():
    examples = [
        ("Interstellar", "Arrival", 4.7),
        ("The Notebook", "A Walk to Remember", 3.9),
    ]
    for inp, rec, rating in examples:
        roast = await llm.generate_roast(inp, rec, rating)
        print(f"INPUT: {inp} -> RECOMMEND: {rec} ({rating})\nROAST: {roast}\n{'-'*60}")

if __name__ == "__main__":
    asyncio.run(main())