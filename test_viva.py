
from agents import viva_agent

print("Testing MUSE Viva Agent...")

try:
    result = viva_agent()
    print("\nVIVA QUESTIONS:\n")
    print(result)

except Exception as e:
    print("\nViva Agent failed:")
    print(type(e).__name__, e)
