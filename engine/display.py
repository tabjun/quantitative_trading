"""노트북 inline 표시 helper.

- `show_figure`, `display_markdown`는 9번에서 옮겼다.
- `display_table`는 12번에서 옮겼다(`max_rows` 인자를 포함한 버전).
"""

from __future__ import annotations

import pandas as pd


def show_figure(fig) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def display_table(title: str, frame: pd.DataFrame, max_rows: int = 40) -> None:
    print(f"\n[{title}]")
    try:
        from IPython.display import display

        display(frame.head(max_rows))
    except Exception:
        print(frame.head(max_rows).to_string(index=False))


def display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except Exception:
        print(text)
