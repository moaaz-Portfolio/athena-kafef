#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef struct {
    bool enabled;
    bool sensor_ok;
    bool distance_valid;
    bool present;
    bool buzzer_active;
    uint16_t distance_cm;
} door_sensor_snapshot_t;

/* HC-SR04: TRIG GPIO4, ECHO GPIO5 via a voltage divider, active buzzer driver GPIO6. */
esp_err_t door_sensor_start(void);
void door_sensor_get(door_sensor_snapshot_t *out);
