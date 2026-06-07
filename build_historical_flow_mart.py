"""Entrypoint for building the KRW-wide historical flow mart.

Run this from the intended server/automation environment, not as a local heavy
research pass during Codex code-generation sessions.
"""

from historical_flow_mart import main


if __name__ == "__main__":
    main()
