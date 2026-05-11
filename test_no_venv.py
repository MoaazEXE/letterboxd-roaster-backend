
import asyncio
import sys
import os

# Add the parent directory of backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import recommend

async def main():
    print("--- Letterboxd Roaster Example (System Python) ---")
    movie = "Inception"
    persona = "casual"
    
    print(f"Input Movie: {movie}")
    print(f"Persona: {persona}")
    print("\nProcessing...\n")
    
    try:
        result = await recommend(movie=movie, persona=persona)
        recommendations = result.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (Rating: {rec['predicted_rating']})")
            print(f"   Roast: {rec['roast']}\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
