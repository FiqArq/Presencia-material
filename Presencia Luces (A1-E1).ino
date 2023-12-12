#include <FastLED.h>

#define Chipset WS2812B
#define Orden_color GRB
#define V 5
#define mA 30000
#define Num_strips 25
#define Num_leds_strip 75
#define Num_leds Num_leds_strip * Num_strips

String EntradaSerial = "";
bool EntradaCompl = false;

struct Zonas {
  float brightness;
  float pulseSpeed;
};

Zonas secs[Num_strips] = {{0}};

CRGB leds[Num_leds];

float VarA = 1, VarB = 255, Var = VarA;
const float delta = (VarB - VarA) / 2.35040238;

void setup() {
  delay(3000);
  FastLED.setMaxPowerInVoltsAndMilliamps(V, mA);
  Serial.begin(9600);

  FastLED.addLeds<Chipset, 3, Orden_color>(leds, 0 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 2, Orden_color>(leds, 1 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 22, Orden_color>(leds, 2 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 24, Orden_color>(leds, 3 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 26, Orden_color>(leds, 4 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 28, Orden_color>(leds, 5 * Num_leds_strip, Num_leds_strip);
  //
  FastLED.addLeds<Chipset, 32, Orden_color>(leds, 6 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 34, Orden_color>(leds, 7 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 36, Orden_color>(leds, 8 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 38, Orden_color>(leds, 9 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 40, Orden_color>(leds, 10 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 42, Orden_color>(leds, 11 * Num_leds_strip, Num_leds_strip);
  //
  FastLED.addLeds<Chipset, 46, Orden_color>(leds, 12 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 48, Orden_color>(leds, 13 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 50, Orden_color>(leds, 14 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 52, Orden_color>(leds, 15 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 5, Orden_color>(leds, 16 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 4, Orden_color>(leds, 17 * Num_leds_strip, Num_leds_strip);
  //
  FastLED.addLeds<Chipset, 25, Orden_color>(leds, 18 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 27, Orden_color>(leds, 19 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 29, Orden_color>(leds, 20 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 31, Orden_color>(leds, 21 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 33, Orden_color>(leds, 22 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 35, Orden_color>(leds, 23 * Num_leds_strip, Num_leds_strip);
  //
  FastLED.addLeds<Chipset, 39, Orden_color>(leds, 24 * Num_leds_strip, Num_leds_strip);
}

void loop() {
  EVERY_N_MILLIS(5) {
    EventoSerial();
    LecturaSerial();
    ActualizacionLeds();
  }
}

void EventoSerial() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    EntradaSerial += inChar;
    if (inChar == '\n') {
      EntradaCompl = true;
    }
  }
}

void LecturaSerial() {
  if (EntradaCompl) {
    String inputs[] = {"a6\n", "a5\n", "a4\n", "a3\n", "a2\n", "a1\n", "b6\n", "b5\n", "b4\n", "b3\n", "b2\n", "b1\n",
    "c6\n", "c5\n", "c4\n", "c3\n", "c2\n", "c1\n", "d6\n", "d5\n", "d4\n", "d3\n", "d2\n", "d1\n", "e6\n"};

    for (int i = 0; i < Num_strips; i++) {
      if (EntradaSerial == inputs[i]) {
        secs[i].brightness += 1.5f;
        secs[i].pulseSpeed += 0.1f;
      }
    }

    if (EntradaSerial == "x\n") {
      for (int i = 0; i < Num_strips; i++) {
        secs[i].brightness -= 1.0f;
        secs[i].pulseSpeed -= 0.02f;
      }
    }

    for (int i = 0; i < Num_strips; i++) {
      secs[i].brightness = constrain(secs[i].brightness, 0, 100);
      secs[i].pulseSpeed = constrain(secs[i].pulseSpeed, 0, 1);
    }

    EntradaSerial = "";
    EntradaCompl = false;
  }
}

void ActualizacionLeds() {
  for (int i = 0; i < Num_strips; i++) {
    float dV = ((exp(sin(secs[i].pulseSpeed * millis() / 2000.0 * PI)) - 0.36787944) * delta);
    Var = VarA + dV;
    for (int j = i * Num_leds_strip; j < (i + 1) * Num_leds_strip; j++) {
      setLed(j, CRGB((round(secs[i].brightness) * Var * 0.01), 0, 0));
    }
  }
  FastLED.show();
}

void setLed(int index, CRGB color) {
  if (index >= 0 && index < Num_leds) {
    leds[index] = color;
  }
}
