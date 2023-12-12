#include <FastLED.h>

#define Chipset WS2812B
#define Orden_color GRB
#define V 5
#define mA 30000


#define Num_strips 5
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
  //
  FastLED.addLeds<Chipset, 9, Orden_color>(leds, 0 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 10, Orden_color>(leds, 1 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 11, Orden_color>(leds, 2 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 12, Orden_color>(leds, 3 * Num_leds_strip, Num_leds_strip);
  FastLED.addLeds<Chipset, 13, Orden_color>(leds, 4 * Num_leds_strip, Num_leds_strip);

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
    String inputs[] = {"e2\n", "e3\n", "e4\n", "e5\n", "e6\n"};

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