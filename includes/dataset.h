#include <stdint.h>

extern uint8_t* train_labels;
extern double*  train_images;
extern uint8_t* test_labels;
extern double*  test_images;

void load_dataset();
