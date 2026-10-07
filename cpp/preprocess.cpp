#include <array>
#include <chrono>
#include <iostream>
#include <string>

struct RequestInput {
    int hour_of_day;
    int day_index;

    int same_category;
    int same_topic;

    double category_match_strength;
    double topic_match_strength;

    double event_category_confidence;
    double promoted_category_confidence;

    double event_topic_confidence;
    double promoted_topic_confidence;

    int user_prior_displays;

    double ad_historical_ctr;
    double campaign_historical_ctr;
    double advertiser_historical_ctr;
    double promoted_document_historical_ctr;

    std::string platform_category;
    std::string country;
    std::string region;
};

struct ProcessedFeatures {
    std::array<double, 15> numeric;

    std::string platform_category;
    std::string country;
    std::string region;
};

ProcessedFeatures preprocess(const RequestInput& input) {
    ProcessedFeatures output;

    output.numeric[0] = static_cast<double>(input.hour_of_day);
    output.numeric[1] = static_cast<double>(input.day_index);
    output.numeric[2] = static_cast<double>(input.same_category);
    output.numeric[3] = static_cast<double>(input.same_topic);

    output.numeric[4] = input.category_match_strength;
    output.numeric[5] = input.topic_match_strength;

    output.numeric[6] = input.event_category_confidence;
    output.numeric[7] = input.promoted_category_confidence;

    output.numeric[8] = input.event_topic_confidence;
    output.numeric[9] = input.promoted_topic_confidence;

    output.numeric[10] =
        static_cast<double>(input.user_prior_displays);

    output.numeric[11] = input.ad_historical_ctr;
    output.numeric[12] = input.campaign_historical_ctr;
    output.numeric[13] = input.advertiser_historical_ctr;
    output.numeric[14] =
        input.promoted_document_historical_ctr;

    output.platform_category =
        input.platform_category;

    output.country =
        input.country;

    output.region =
        input.region;

    return output;
}

int main() {
    RequestInput request{
        14,     // hour_of_day
        3,      // day_index
        1,      // same_category
        0,      // same_topic

        0.82,   // category_match_strength
        0.0,    // topic_match_strength

        0.91,   // event_category_confidence
        0.90,   // promoted_category_confidence

        0.78,   // event_topic_confidence
        0.65,   // promoted_topic_confidence

        12,     // user_prior_displays

        0.21,   // ad_historical_ctr
        0.18,   // campaign_historical_ctr
        0.16,   // advertiser_historical_ctr
        0.24,   // promoted_document_historical_ctr

        "1",    // platform_category
        "US",   // country
        "CA"    // region
    };

    const int iterations = 100000;

    // Warm-up
    ProcessedFeatures result;

    for (int i = 0; i < 10000; ++i) {
        result = preprocess(request);
    }

    double checksum = 0.0;

    auto start =
        std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        result = preprocess(request);

        // Prevent compiler from optimizing away the loop.
        checksum += result.numeric[0];
    }

    auto end =
        std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<
            std::chrono::nanoseconds>(
            end - start
        );

    double total_us =
        static_cast<double>(duration.count())
        / 1000.0;

    double average_us =
        total_us
        / static_cast<double>(iterations);

    std::cout
        << "C++ preprocessing benchmark\n";

    std::cout
        << "Iterations: "
        << iterations
        << "\n";

    std::cout
        << "Total time: "
        << total_us
        << " us\n";

    std::cout
        << "Average per request: "
        << average_us
        << " us\n";

    std::cout
        << "Checksum: "
        << checksum
        << "\n";

    std::cout
        << "\nExample output:\n";

    const char* numeric_names[15] = {
        "hour_of_day",
        "day_index",
        "same_category",
        "same_topic",
        "category_match_strength",
        "topic_match_strength",
        "event_category_confidence",
        "promoted_category_confidence",
        "event_topic_confidence",
        "promoted_topic_confidence",
        "user_prior_displays",
        "ad_historical_ctr",
        "campaign_historical_ctr",
        "advertiser_historical_ctr",
        "promoted_document_historical_ctr"
    };

    for (int i = 0; i < 15; ++i) {
        std::cout
            << numeric_names[i]
            << ": "
            << result.numeric[i]
            << "\n";
    }

    std::cout
        << "platform_category: "
        << result.platform_category
        << "\n";

    std::cout
        << "country: "
        << result.country
        << "\n";

    std::cout
        << "region: "
        << result.region
        << "\n";

    return 0;
}