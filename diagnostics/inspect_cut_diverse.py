import argparse
import json


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(cut_file, trace_file, n_per_group):
    cuts = load_rows(cut_file)
    traces = load_rows(trace_file)

    aligned = len(cuts) == len(traces) and all(c["id"] == t["id"] for c, t in zip(cuts, traces))
    if not aligned:
        print("WARNING: cut file and trace file do not line up row-for-row; "
              "final predicted answers will not be shown.\n")

    picked = {"incorrect": [], "correct": []}
    seen_ids = set()
    # incorrect first, one example per problem, so different problems show up
    for label in ["incorrect", "correct"]:
        want_correct = label == "correct"
        for i, c in enumerate(cuts):
            if len(picked[label]) >= n_per_group:
                break
            if c.get("never_explicit") or c["correct"] != want_correct:
                continue
            if c["id"] in seen_ids:
                continue
            seen_ids.add(c["id"])
            picked[label].append(i)

    for label in ["incorrect", "correct"]:
        print("#" * 78)
        print(f"{label.upper()} traces (one per problem, cut = after first sentence containing the gold answer)")
        print("#" * 78)
        for i in picked[label]:
            c = cuts[i]
            print("=" * 78)
            print(f"id={c['id']}  cut_target={c.get('cut_target')}  gold={c['answer']}  frac_words_kept={c.get('frac_words_kept')}")
            if aligned:
                print(f"model's final answer: {traces[i].get('predicted_answer')}")
            print("-" * 78)
            print("END OF KEPT TEXT:")
            print(c["adaptive_truncated_text"][-250:])
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--n-per-group", type=int, default=4)
    args = parser.parse_args()
    main(args.cut_file, args.trace_file, args.n_per_group)
