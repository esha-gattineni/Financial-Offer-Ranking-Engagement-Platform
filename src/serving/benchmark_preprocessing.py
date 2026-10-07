import time


def preprocess(request):
    numeric = {
        "hour_of_day": float(request["hour_of_day"]),
        "day_index": float(request["day_index"]),
        "same_category": float(request["same_category"]),
        "same_topic": float(request["same_topic"]),
        "category_match_strength": request["category_match_strength"],
        "topic_match_strength": request["topic_match_strength"],
        "event_category_confidence": request["event_category_confidence"],
        "promoted_category_confidence": request["promoted_category_confidence"],
        "event_topic_confidence": request["event_topic_confidence"],
        "promoted_topic_confidence": request["promoted_topic_confidence"],
        "user_prior_displays": float(request["user_prior_displays"]),
        "ad_historical_ctr": request["ad_historical_ctr"],
        "campaign_historical_ctr": request["campaign_historical_ctr"],
        "advertiser_historical_ctr": request["advertiser_historical_ctr"],
        "promoted_document_historical_ctr":
            request["promoted_document_historical_ctr"],
    }

    categorical = {
        "platform_category": request["platform_category"],
        "country": request["country"],
        "region": request["region"],
    }

    return numeric, categorical


def main():
    request = {
        "hour_of_day": 14,
        "day_index": 3,
        "same_category": 1,
        "same_topic": 0,
        "category_match_strength": 0.82,
        "topic_match_strength": 0.0,
        "event_category_confidence": 0.91,
        "promoted_category_confidence": 0.90,
        "event_topic_confidence": 0.78,
        "promoted_topic_confidence": 0.65,
        "user_prior_displays": 12,
        "ad_historical_ctr": 0.21,
        "campaign_historical_ctr": 0.18,
        "advertiser_historical_ctr": 0.16,
        "promoted_document_historical_ctr": 0.24,
        "platform_category": "1",
        "country": "US",
        "region": "CA",
    }

    iterations = 100_000

    # Warm-up
    for _ in range(10_000):
        preprocess(request)

    start = time.perf_counter_ns()

    for _ in range(iterations):
        result = preprocess(request)

    end = time.perf_counter_ns()

    total_ns = end - start

    total_us = total_ns / 1000
    average_us = total_us / iterations

    print("Python preprocessing benchmark")
    print(f"Iterations: {iterations:,}")
    print(f"Total time: {total_us:.0f} us")
    print(f"Average per request: {average_us:.4f} us")

    print("\nExample output:")
    print(result)


if __name__ == "__main__":
    main()