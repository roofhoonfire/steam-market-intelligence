import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "gamalytic"
)

GAMALYTIC_URL = (
    "https://api.gamalytic.com/game/{appid}"
)


def collect_game(appid):

    url = GAMALYTIC_URL.format(
        appid=appid
    )

    print(
        f"[Gamalytic] requesting "
        f"appid={appid}"
    )

    response = requests.get(
        url,
        timeout=30,
    )

    print(
        "HTTP status:",
        response.status_code
    )

    print("Response body:")
    print(response.text[:1000])
    response.raise_for_status()

    payload = response.json()

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RAW_DIR
        / f"{appid}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved: {output_path}"
    )

    print()
    print("Top-level fields:")

    if isinstance(payload, dict):

        for key in payload.keys():
            print(" -", key)

    else:
        print(
            "Unexpected payload type:",
            type(payload)
        )


if __name__ == "__main__":

    # Meg's Monster
    collect_game(
        appid=1783360
    )
