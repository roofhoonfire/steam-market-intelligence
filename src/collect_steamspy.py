import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "steamspy"
)

STEAMSPY_URL = "https://steamspy.com/api.php"


def collect_game(appid):

    print(
        f"[SteamSpy] requesting "
        f"appid={appid}"
    )

    params = {
        "request": "appdetails",
        "appid": appid,
    }

    response = requests.get(
        STEAMSPY_URL,
        params=params,
        timeout=30,
    )

    print(
        "HTTP status:",
        response.status_code
    )

    # 에러가 나면 원인 확인용
    if response.status_code != 200:
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
    print("=== SteamSpy summary ===")

    interesting_fields = [
        "appid",
        "name",
        "developer",
        "publisher",
        "positive",
        "negative",
        "owners",
        "average_forever",
        "median_forever",
        "price",
        "initialprice",
        "discount",
        "ccu",
    ]

    for key in interesting_fields:
        print(
            f"{key:20}: "
            f"{payload.get(key)}"
        )

    print()
    print("tags:")

    tags = payload.get(
        "tags",
        {}
    )

    if isinstance(tags, dict):

        for tag, count in list(
            tags.items()
        )[:15]:
            print(
                f"  {tag}: {count}"
            )


if __name__ == "__main__":

    # Meg's Monster
    collect_game(
        appid=1783360
    )
