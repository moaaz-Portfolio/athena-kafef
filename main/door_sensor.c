#include "door_sensor.h"

#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "settings.h"

#define DOOR_TRIG_GPIO GPIO_NUM_4
#define DOOR_ECHO_GPIO GPIO_NUM_5
#define BUZZER_GPIO GPIO_NUM_6
#define ECHO_TIMEOUT_US 30000
#define SAMPLE_DELAY_MS 80
#define UDP_PACKET_INTERVAL_MS 500

static const char *TAG = "door_sensor";
static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static door_sensor_snapshot_t s_state;
static bool s_started;

typedef struct __attribute__((packed)) {
    char magic[4];
    uint32_t sequence;
    uint16_t distance_cm;
    uint8_t present;
    uint8_t sensor_ok;
    uint8_t buzzer_active;
} door_packet_t;

void door_sensor_get(door_sensor_snapshot_t *out)
{
    if (!out) return;
    portENTER_CRITICAL(&s_lock);
    *out = s_state;
    portEXIT_CRITICAL(&s_lock);
}

static bool read_distance(uint16_t *distance_cm)
{
    gpio_set_level(DOOR_TRIG_GPIO, 0);
    esp_rom_delay_us(3);
    gpio_set_level(DOOR_TRIG_GPIO, 1);
    esp_rom_delay_us(10);
    gpio_set_level(DOOR_TRIG_GPIO, 0);

    /* Time out separately while waiting for the rising edge and pulse end. */
    int64_t rise_deadline = esp_timer_get_time() + ECHO_TIMEOUT_US;
    while (gpio_get_level(DOOR_ECHO_GPIO) == 0) {
        if (esp_timer_get_time() >= rise_deadline) return false;
    }

    int64_t pulse_start = esp_timer_get_time();
    int64_t pulse_deadline = pulse_start + ECHO_TIMEOUT_US;
    while (gpio_get_level(DOOR_ECHO_GPIO) == 1) {
        if (esp_timer_get_time() >= pulse_deadline) return false;
    }

    int64_t pulse_us = esp_timer_get_time() - pulse_start;
    if (pulse_us <= 0 || pulse_us > ECHO_TIMEOUT_US) return false;

    float cm = ((float)pulse_us * 0.0343f) / 2.0f;
    if (cm < 2.0f || cm > 400.0f) return false;
    *distance_cm = (uint16_t)(cm + 0.5f);
    return true;
}

static void publish_udp(int sock, uint32_t sequence, const door_sensor_snapshot_t *state)
{
    if (sock < 0) return;

    const settings_t *cfg = settings_get();
    uint16_t stream_port = cfg->stream_port ? cfg->stream_port : 5005;
    uint16_t telemetry_port = stream_port < 65535 ? (uint16_t)(stream_port + 1) : 5006;

    struct sockaddr_in dest = {0};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(telemetry_port);
    if (cfg->stream_host[0] == 0 ||
        inet_pton(AF_INET, cfg->stream_host, &dest.sin_addr) != 1) {
        dest.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    }

    door_packet_t packet = {
        .magic = {'D', 'O', 'R', '1'},
        .sequence = sequence,
        .distance_cm = state->distance_valid ? state->distance_cm : 0,
        .present = state->present ? 1 : 0,
        .sensor_ok = state->sensor_ok ? 1 : 0,
        .buzzer_active = state->buzzer_active ? 1 : 0,
    };
    (void)sendto(sock, &packet, sizeof(packet), 0,
                 (struct sockaddr *)&dest, sizeof(dest));
}

static void door_task(void *arg)
{
    (void)arg;
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock >= 0) {
        int enabled = 1;
        (void)setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &enabled, sizeof(enabled));
    }

    uint32_t sequence = 0;
    uint32_t tick = 0;
    uint32_t last_publish_tick = UINT32_MAX;
    unsigned near_count = 0;
    unsigned far_count = 0;
    bool present = false;

    for (;;) {
        const settings_t *cfg = settings_get();
        door_sensor_snapshot_t next = {0};
        next.enabled = cfg->door_enabled;
        next.sensor_ok = cfg->door_enabled;

        uint16_t distance = 0;
        bool valid = cfg->door_enabled && read_distance(&distance);
        next.distance_valid = valid;
        next.distance_cm = valid ? distance : 0;

        if (!cfg->door_enabled) {
            near_count = 0;
            far_count = 0;
            present = false;
        } else if (valid && distance <= cfg->door_threshold_cm) {
            near_count++;
            far_count = 0;
            if (near_count >= 2) present = true;
        } else {
            near_count = 0;
            far_count++;
            if (far_count >= 3) present = false;
        }

        next.present = present;
        next.buzzer_active = cfg->door_enabled && cfg->buzzer_enabled &&
                             present && (tick % 9 < 2);
        gpio_set_level(BUZZER_GPIO, next.buzzer_active ? 1 : 0);

        portENTER_CRITICAL(&s_lock);
        s_state = next;
        portEXIT_CRITICAL(&s_lock);

        if (last_publish_tick == UINT32_MAX ||
            (tick - last_publish_tick) * SAMPLE_DELAY_MS >= UDP_PACKET_INTERVAL_MS) {
            publish_udp(sock, sequence++, &next);
            last_publish_tick = tick;
        }

        tick++;
        vTaskDelay(pdMS_TO_TICKS(SAMPLE_DELAY_MS));
    }
}

esp_err_t door_sensor_start(void)
{
    if (s_started) return ESP_ERR_INVALID_STATE;

    gpio_config_t trig = {
        .pin_bit_mask = 1ULL << DOOR_TRIG_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t err = gpio_config(&trig);
    if (err != ESP_OK) return err;
    gpio_set_level(DOOR_TRIG_GPIO, 0);

    gpio_config_t echo = {
        .pin_bit_mask = 1ULL << DOOR_ECHO_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    err = gpio_config(&echo);
    if (err != ESP_OK) return err;

    gpio_config_t buzzer = {
        .pin_bit_mask = 1ULL << BUZZER_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    err = gpio_config(&buzzer);
    if (err != ESP_OK) return err;
    gpio_set_level(BUZZER_GPIO, 0);

    s_started = true;
    BaseType_t task_ok = xTaskCreate(door_task, "door_sensor", 4096, NULL, 3, NULL);
    if (task_ok != pdPASS) {
        s_started = false;
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "HC-SR04 TRIG=%d ECHO=%d, buzzer driver=%d",
             DOOR_TRIG_GPIO, DOOR_ECHO_GPIO, BUZZER_GPIO);
    return ESP_OK;
}
