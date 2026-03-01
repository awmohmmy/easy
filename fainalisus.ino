#include <FastLED.h>

#define BAUDRATE 115200
#define NUM_LEDS 61
#define BRIGHTNESS 80

// ===== Lane A =====
#define PIN_GREEN_A 21
#define PIN_RED_A   23

// ===== Lane B =====
#define PIN_GREEN_B 18
#define PIN_RED_B   19

#define LED_TYPE WS2812B
#define COLOR_ORDER GRB

CRGB ledsGreenA[NUM_LEDS];
CRGB ledsRedA[NUM_LEDS];

CRGB ledsGreenB[NUM_LEDS];
CRGB ledsRedB[NUM_LEDS];

void setup() {
  Serial.begin(BAUDRATE);

  FastLED.addLeds<LED_TYPE, PIN_GREEN_A, COLOR_ORDER>(ledsGreenA, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, PIN_RED_A,   COLOR_ORDER>(ledsRedA, NUM_LEDS);

  FastLED.addLeds<LED_TYPE, PIN_GREEN_B, COLOR_ORDER>(ledsGreenB, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, PIN_RED_B,   COLOR_ORDER>(ledsRedB, NUM_LEDS);

  FastLED.setBrightness(BRIGHTNESS);
  allOff();

  Serial.println("ESP32 READY");
}

void loop() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "GREEN_A") {
    laneA_Green();
    Serial.println("OK GREEN_A");
  }
  else if (cmd == "RED_A") {
    laneB_Green();
    Serial.println("OK GREEN_B");
  }
  else if (cmd == "OFF") {
    allOff();
    Serial.println("OK OFF");
  }
}

// ===== Lane A เขียว =====
void laneA_Green() {
  for (int i = 0; i < NUM_LEDS; i++) {

    // Lane A เขียว
    ledsGreenA[i] = CRGB::Green;
    ledsRedA[i]   = CRGB::Black;

    // Lane B แดง
    ledsGreenB[i] = CRGB::Black;
    ledsRedB[i]   = CRGB::Red;
  }
  FastLED.show();
}

// ===== Lane B เขียว =====
void laneB_Green() {
  for (int i = 0; i < NUM_LEDS; i++) {

    // Lane A แดง
    ledsGreenA[i] = CRGB::Black;
    ledsRedA[i]   = CRGB::Red;

    // Lane B เขียว
    ledsGreenB[i] = CRGB::Green;
    ledsRedB[i]   = CRGB::Black;
  }
  FastLED.show();
}

void allOff() {
  for (int i = 0; i < NUM_LEDS; i++) {

    ledsGreenA[i] = CRGB::Black;
    ledsRedA[i]   = CRGB::Black;

    ledsGreenB[i] = CRGB::Black;
    ledsRedB[i]   = CRGB::Black;
  }
  FastLED.show();
}