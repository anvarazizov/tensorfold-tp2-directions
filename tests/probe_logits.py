"""Issue the four cold-prefill requests whose logits are compared in RESULTS.md."""
import argparse
import json
import os
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017-runtime-directions-tp2")
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TF_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TF_API_KEY"]
    for strength in (None, 0, 0.5, None):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Return only the integer: 31 * 17."}],
            "temperature": 0, "seed": 42, "max_tokens": 1, "reasoning_effort": "none",
            "cache_salt": uuid.uuid4().hex,
        }
        if strength is not None:
            body["ablation"] = {"strength": strength, "enabled": strength > 0,
                                "direction_set": "pqhaz-1017-projected-experimental"}
        request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            print(strength, json.load(response)["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
