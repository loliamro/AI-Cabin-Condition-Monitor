#include <Arduino_LSM9DS1.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h"
#include "scaler.h"
#include "labels.h"

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int thermistorPin = A0;
const int ldrPin = A1;
const int buzzerPin = 9;  // NEW

const float VCC = 3.3;
const float ADC_MAX = 4095.0;

const float fixedResistor = 22000.0;
const float nominalResistance = 47000.0;
const float nominalTemperature = 25.0;
const float betaCoefficient = 3950.0;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 48 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

float windowBuf[WINDOW_SIZE][NUM_FEATURES];
int windowHead = 0;
int windowCount = 0;

float prevAccX = 0, prevAccY = 0, prevAccZ = 0;
bool firstAccelRead = true;

unsigned long lastLCDUpdate = 0;

void showLCD(const char* l1, const char* l2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(l1);
  lcd.setCursor(0, 1);
  lcd.print(l2);
}

float resistanceToTemp(float resistance) {
  float tempK = resistance / nominalResistance;
  tempK = log(tempK);
  tempK /= betaCoefficient;
  tempK += 1.0 / (nominalTemperature + 273.15);
  tempK = 1.0 / tempK;
  return tempK - 273.15;
}

float readTemperatureC() {
  long total = 0;

  for (int i = 0; i < 20; i++) {
    total += analogRead(thermistorPin);
    delay(2);
  }

  int raw = total / 20;

  if (raw < 5 || raw > 4090) {
    Serial.print("THERMISTOR BAD RAW: ");
    Serial.println(raw);
    return -999;
  }

  float voltage = raw * (VCC / ADC_MAX);
  float thermistorResistance = fixedResistor * ((VCC - voltage) / voltage);
  float tempC = resistanceToTemp(thermistorResistance);

  Serial.print("RAW: ");
  Serial.print(raw);
  Serial.print("  V: ");
  Serial.print(voltage);
  Serial.print("  R: ");
  Serial.print(thermistorResistance);
  Serial.print("  T: ");
  Serial.println(tempC);

  if (tempC < -20 || tempC > 120) {
    Serial.println("THERMISTOR TEMP OUT OF RANGE");
    return -999;
  }

  return tempC;
}

int readLDRRaw() {
  long total = 0;

  for (int i = 0; i < 20; i++) {
    total += analogRead(ldrPin);
    delay(2);
  }

  int raw12 = total / 20;
  int raw10 = (int)(raw12 * (1023.0 / 4095.0));

  return raw10;
}

float readMovementScore() {
  float accX, accY, accZ;

  if (!IMU.accelerationAvailable()) return 0.0;

  IMU.readAcceleration(accX, accY, accZ);

  if (firstAccelRead) {
    prevAccX = accX;
    prevAccY = accY;
    prevAccZ = accZ;
    firstAccelRead = false;
    return 0.0;
  }

  float dx = accX - prevAccX;
  float dy = accY - prevAccY;
  float dz = accZ - prevAccZ;

  prevAccX = accX;
  prevAccY = accY;
  prevAccZ = accZ;

  return sqrt(dx * dx + dy * dy + dz * dz);
}

float scaleFeature(float value, int index) {
  return (value - featureMeans[index]) / featureScales[index];
}

void addToWindow(float tempC, float ldrRaw, float movementScore) {
  int writeIdx;

  if (windowCount < WINDOW_SIZE) {
    writeIdx = windowCount;
    windowCount++;
  } else {
    writeIdx = windowHead;
    windowHead = (windowHead + 1) % WINDOW_SIZE;
  }

  windowBuf[writeIdx][0] = scaleFeature(tempC, 0);
  windowBuf[writeIdx][1] = scaleFeature(ldrRaw, 1);
  windowBuf[writeIdx][2] = scaleFeature(movementScore, 2);
}

void fillModelInput() {
  int index = 0;

  for (int t = 0; t < WINDOW_SIZE; t++) {
    int src = (windowCount < WINDOW_SIZE) ? t : (windowHead + t) % WINDOW_SIZE;

    for (int f = 0; f < NUM_FEATURES; f++) {
      input->data.f[index++] = windowBuf[src][f];
    }
  }
}

void updateLCDPrediction(const char* condition, float confidence, float tempC, int ldrRaw) {
  if (millis() - lastLCDUpdate < 700) return;

  lastLCDUpdate = millis();

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print(condition);
  lcd.print(" ");
  lcd.print((int)(confidence * 100));
  lcd.print("%");

  lcd.setCursor(0, 1);
  lcd.print("T:");
  lcd.print(tempC, 1);
  lcd.print(" L:");
  lcd.print(ldrRaw);
}

// NEW: buzzer beep pattern based on condition
void handleBuzzer(int predictedIndex) {
  if (predictedIndex == 2) {
    // Alert: 3 short beeps
    for (int i = 0; i < 3; i++) {
      digitalWrite(buzzerPin, HIGH);
      delay(150);
      digitalWrite(buzzerPin, LOW);
      delay(100);
    }
  } else if (predictedIndex == 1) {
    // Caution: 1 short beep
    digitalWrite(buzzerPin, HIGH);
    delay(200);
    digitalWrite(buzzerPin, LOW);
  } else {
    // Normal: silence
    digitalWrite(buzzerPin, LOW);
  }
}

void setup() {
  Serial.begin(9600);
  delay(1000);

  analogReadResolution(12);

  pinMode(buzzerPin, OUTPUT);      // NEW
  digitalWrite(buzzerPin, LOW);    // NEW: ensure buzzer starts off

  Wire.begin();
  lcd.init();
  lcd.backlight();

  showLCD("Cabin Monitor", "Starting...");
  delay(1000);

  if (!IMU.begin()) {
    showLCD("ERROR", "IMU failed");
    while (1);
  }

  showLCD("Loading model", "Please wait");
  delay(500);

  model = tflite::GetModel(cabin_cnn_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    showLCD("ERROR", "Model schema");
    while (1);
  }

  static tflite::MicroErrorReporter microErrorReporter;
  static tflite::ErrorReporter* errorReporter = &microErrorReporter;

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();

  static tflite::MicroInterpreter staticInterpreter(
    model, resolver, tensorArena, kTensorArenaSize, errorReporter, nullptr);

  interpreter = &staticInterpreter;

  showLCD("Alloc tensors", "Please wait");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    showLCD("ERROR", "Tensor fail");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  showLCD("TinyML Ready", "Filling data");
  delay(1000);
}

void loop() {
  float tempC = readTemperatureC();
  int ldrRaw = readLDRRaw();
  float movementScore = readMovementScore();

  if (tempC == -999) {
    showLCD("Sensor error", "Thermistor");
    delay(500);
    return;
  }

  addToWindow(tempC, (float)ldrRaw, movementScore);

  if (windowCount < WINDOW_SIZE) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Filling window");
    lcd.setCursor(0, 1);
    lcd.print(windowCount);
    lcd.print("/");
    lcd.print(WINDOW_SIZE);
    delay(300);
    return;
  }

  fillModelInput();

  if (interpreter->Invoke() != kTfLiteOk) {
    showLCD("ERROR", "Invoke failed");
    delay(500);
    return;
  }

  float pNormal = output->data.f[0];
  float pCaution = output->data.f[1];
  float pAlert = output->data.f[2];

  int predictedIndex = 0;
  float confidence = pNormal;

  if (pCaution > confidence) {
    predictedIndex = 1;
    confidence = pCaution;
  }

  if (pAlert > confidence) {
    predictedIndex = 2;
    confidence = pAlert;
  }

  const char* prediction = CLASS_NAMES[predictedIndex];

  updateLCDPrediction(prediction, confidence, tempC, ldrRaw);
  handleBuzzer(predictedIndex);  // NEW

  delay(300);
}