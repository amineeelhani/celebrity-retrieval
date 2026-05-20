import json
import requests


def submit(results, groupname, url):
    """
    Send results to the evaluation server.

    Args:
        results (dict): { "query_img.jpg": ["gallery1.jpg", ..., "gallery10.jpg"], ... }
        groupname (str): your group name
        url (str): server URL (e.g. "http://localhost:3001/retrieval/")
    """
    payload = {
        "groupname": groupname,
        "images": results
    }
    response = requests.post(url, json.dumps(payload))
    try:
        result = json.loads(response.text)
        print(f"Server response: {result}")
        print(f"accuracy is {result['accuracy']}")
        return result
    except json.JSONDecodeError:
        print(f"Error: {response.text}")
        return None


def evaluate_local(results, ground_truth):
    """
    Evaluate results locally (no server needed) for debugging.

    Args:
        results (dict): { "query_img.jpg": ["g1.jpg", ..., "g10.jpg"] }
        ground_truth (dict): { "query_img.jpg": "correct_gallery_img.jpg" }

    Returns:
        dict with top1, top5, top10 accuracy and total score
    """
    top1, top5, top10 = 0, 0, 0
    n = len(results)

    for query, ranked in results.items():
        correct = ground_truth.get(query)
        if correct is None:
            continue
        # supporta sia stringa che lista
        if isinstance(correct, str):
            correct = [correct]
        if ranked[0] in correct:
            top1 += 1
        if any(c in ranked[:5] for c in correct):
            top5 += 1
        if any(c in ranked[:10] for c in correct):
            top10 += 1

    acc1 = top1 / n
    acc5 = top5 / n
    acc10 = top10 / n
    score = acc1 * 600 + acc5 * 300 + acc10 * 100

    print(f"Top-1:  {acc1*100:.1f}%")
    print(f"Top-5:  {acc5*100:.1f}%")
    print(f"Top-10: {acc10*100:.1f}%")
    print(f"Score:  {score:.1f} / 1000")

    return {"top1": acc1, "top5": acc5, "top10": acc10, "score": score}
