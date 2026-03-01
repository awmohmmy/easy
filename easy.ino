#include <FastLED.h>

#define BAUDRATE 115200
#define NUM_LEDS 61
#define BRIGHTNESS 80

#define PIN_GREEN 21
#define PIN_RED   23

#define LED_TYPE WS2812B
#define COLOR_ORDER GRB

CRGB ledsGreen[NUM_LEDS];
CRGB ledsRed[NUM_LEDS];

void setup() {
  Serial.begin(BAUDRATE);

  FastLED.addLeds<LED_TYPE, PIN_GREEN, COLOR_ORDER>(ledsGreen, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, PIN_RED,   COLOR_ORDER>(ledsRed,   NUM_LEDS);

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
    greenOn();
    Serial.println("OK GREEN_A");
  }
  else if (cmd == "RED_A") {
    redOn();
    Serial.println("OK RED_A");
  }
  else if (cmd == "OFF") {
    allOff();
    Serial.println("OK OFF");
  }
}

void greenOn() {
  for (int i = 0; i < NUM_LEDS; i++) {
    ledsGreen[i] = CRGB::Green;
    ledsRed[i]   = CRGB::Black;
  }
  FastLED.show();
}

void redOn() {
  for (int i = 0; i < NUM_LEDS; i++) {
    ledsRed[i]   = CRGB::Red;
    ledsGreen[i] = CRGB::Black;
  }
  FastLED.show();
}

void allOff() {
  for (int i = 0; i < NUM_LEDS; i++) {
    ledsGreen[i] = CRGB::Black;
    ledsRed[i]   = CRGB::Black;
  }
  FastLED.show();
}
