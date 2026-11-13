
from agents import (
    summarizer_agent,
    concept_extractor_agent,
    slide_generator_agent
)

agents = {
    "SUMMARIZER": summarizer_agent,
    "CONCEPT EXTRACTOR": concept_extractor_agent,
    "SLIDE GENERATOR": slide_generator_agent
}

for name, agent in agents.items():

    print(f"\n===== {name} AGENT =====\n")

    try:
        result = agent()
        print(result)

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

print("\n===== TESTING COMPLETED =====")
