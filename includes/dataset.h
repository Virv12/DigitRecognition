#include <stdint.h>
#include <array>
#include <vector>

extern std::vector<uint8_t> train_labels;
extern std::vector<std::array<uint8_t, 28*28>> train_images;
extern std::vector<uint8_t> test_labels;
extern std::vector<std::array<uint8_t, 28*28>> test_images;

void load_dataset();
