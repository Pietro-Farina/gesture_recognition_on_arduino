/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/

#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <arm_math.h>
#include "model.h"

// --- Constants ---
const int N = 128;     // Must be power of 2 for ARM FFT
const int numAxes = 6; // aX, aY, aZ, gX, gY, gZ
const float fs = 62.5; // sampling rate
const int SAMPLES_PER_GESTURE = 125;

// --- Buffers & FFT Global Variables ---
float doubleBuffers[numAxes][2 * N];
int writeIndex = 0;
int newSamplesCount = 0;
bool isCapturing = false;

arm_rfft_fast_instance_f32 fft_inst;
float fftOutput[N];
float magBuffer[N / 2];

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char *GESTURES[] = {
    "punch",
    "updown",
    "shake",
    "rest",
    "circle"};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup()
{
  Serial.begin(9600);
  while (!Serial)
    ;

  // initialize the IMU
  if (!IMU.begin())
  {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Initialize fft
  arm_rfft_fast_init_f32(&fft_inst, N);
}

void loop()
{
  float rawData[numAxes];

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
  {
    float aX, aY, aZ, gX, gY, gZ;
    IMU.readAcceleration(aX, aY, aZ);
    IMU.readGyroscope(gX, gY, gZ);

    // Normalize input
    rawData[0] = (aX + 4.0) / 8.0;
    rawData[1] = (aY + 4.0) / 8.0;
    rawData[2] = (aZ + 4.0) / 8.0;
    rawData[3] = (gX + 2000.0) / 4000.0;
    rawData[4] = (gY + 2000.0) / 4000.0;
    rawData[5] = (gZ + 2000.0) / 4000.0;

    // 1. Fill Double Buffers with raw data
    for (int i = 0; i < numAxes; i++)
    {
      doubleBuffers[i][writeIndex] = rawData[i];
      doubleBuffers[i][writeIndex + N] = rawData[i];
    }

    writeIndex++;
    newSamplesCount++;

    // 2. Window (125 samples) is Ready
    if (newSamplesCount >= SAMPLES_PER_GESTURE)
    {
      // --- ADD ZERO PADDING TO REACH N=128 ---
      // We fill the 3 "gap" slots in the double buffer with 0.0
      for (int p = 0; p < (N - SAMPLES_PER_GESTURE); p++)
      {
        int padIdx = (writeIndex + p) % N;
        for (int i = 0; i < numAxes; i++)
        {
          doubleBuffers[i][padIdx] = 0.0;
          doubleBuffers[i][padIdx + N] = 0.0;
        }
      }

      int featureIdx = 0;

      for (int i = 0; i < numAxes; i++)
      {
        // The window now effectively contains 125 samples + 3 zeros
        float *window = &doubleBuffers[i][writeIndex - SAMPLES_PER_GESTURE];

        // A. Time Domain Features (Calculated on the full N=128)
        float mean, stdDev, minVal, maxVal, rms;
        uint32_t minIdx, maxIdx;

        arm_mean_f32(window, N, &mean);
        arm_std_f32(window, N, &stdDev);
        arm_min_f32(window, N, &minVal, &minIdx);
        arm_max_f32(window, N, &maxVal, &maxIdx);
        arm_rms_f32(window, N, &rms);

        tflInputTensor->data.f[featureIdx++] = mean;
        tflInputTensor->data.f[featureIdx++] = stdDev;
        tflInputTensor->data.f[featureIdx++] = rms;
        tflInputTensor->data.f[featureIdx++] = minVal;
        tflInputTensor->data.f[featureIdx++] = maxVal;

        // B. Frequency Domain Features (FFT on detrended buffer)
        float detrended[N];
        arm_offset_f32(window, -mean, detrended, N);

        // Keep original buffer clean
        arm_rfft_fast_f32(&fft_inst, detrended, fftOutput, 0);
        arm_cmplx_mag_f32(fftOutput, magBuffer, N / 2);

        // Convert to PSD
        for (int k = 0; k < N / 2; k++)
        {
          magBuffer[k] = (magBuffer[k] * magBuffer[k]) / (N * fs);
        }

        // Top 3 Peaks
        float tempMag[N / 2];
        memcpy(tempMag, magBuffer, sizeof(tempMag));
        tempMag[0] = 0;

        // Skip DC
        for (int p = 0; p < 3; p++)
        {
          float pAmp;
          uint32_t pIdx;
          arm_max_f32(tempMag, N / 2, &pAmp, &pIdx);
          tflInputTensor->data.f[featureIdx++] = pAmp;
          tflInputTensor->data.f[featureIdx++] = (pIdx * fs) / N;

          tempMag[pIdx] = 0;
          if (pIdx > 0)
            tempMag[pIdx - 1] = 0;
          if (pIdx < (N / 2) - 1)
            tempMag[pIdx + 1] = 0;
        }

        // 4 Spectral Bins
        int binSize = (N / 2) / 4;
        for (int b = 0; b < 4; b++)
        {
          float energy = 0;
          for (int j = 0; j < binSize; j++)
            energy += magBuffer[b * binSize + j];

          tflInputTensor->data.f[featureIdx++] = energy;
        }
      }

      // 3. Inference
      if (tflInterpreter->Invoke() == kTfLiteOk)
      {
        for (int i = 0; i < NUM_GESTURES; i++)
        {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 3);
        }
        Serial.println("---");
      }
      // Clean up and reset
      writeIndex = 0;
      newSamplesCount = 0;
    }
    else
    {
      delay(16); // to reach 125 samples each second
    }
  }
}