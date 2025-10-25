#include <Servo.h>

static const int N = 4;
static const uint8_t PINS[N] = {9, 10, 6, 5};
Servo sv[N];

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N; ++i) { sv[i].attach(PINS[i]); sv[i].write(90); }
  Serial.println("READY");
}

void loop() {
  static char buf[32]; static uint8_t i = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      buf[i] = 0; i = 0;
      int idx = -1, us = -1, deg = -1;
      if (sscanf(buf, "S%d=us%d", &idx, &us) == 2) {
        if (0 <= idx && idx < N) {
          // sv[idx].writeMicroseconds(us);
          Serial.println(us);
        }
      } else if (sscanf(buf, "S%d=%d", &idx, &deg) == 2) {
        if (0 <= idx && idx < N) {
          if (deg < 0) deg = 0; if (deg > 180) deg = 180;
          // sv[idx].write(deg);
          Serial.println(deg);
        }
      }
      if (sscanf(buf, "S%d=us%d", &idx, &us) == 2) {
        if (0 <= idx && idx < N) {
          sv[idx].writeMicroseconds(us);
          Serial.print("ACK S"); 
          Serial.print(idx); 
          Serial.print(" us="); 
          Serial.println(us);
        }
      } else if (sscanf(buf, "S%d=%d", &idx, &deg) == 2) {
        if (0 <= idx && idx < N) {
          if (deg < 0) deg = 0; 
          if (deg > 180) deg = 180;
          sv[idx].write(deg);
          Serial.print("ACK S"); 
          Serial.print(idx); 
          Serial.print(" deg="); 
          Serial.println(deg);
        }
      }   
    } else if (i + 1 < sizeof(buf)) buf[i++] = c; 
  }
}
