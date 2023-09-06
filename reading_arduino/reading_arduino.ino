/*
  Reads all 9 Channels from Sensors, and writes to Serial Port repeatedly ~400-500 samples/second.
  Turns on/off vibration motor based on serial input.
*/

IntervalTimer myTimer;

bool collect_data = false;

const int MOTOR_PIN = 4;
bool motor_on = false;

int vib_counter = 0;
const int VIB = 30;
int MAX_VIB_COUNT = VIB*2;`

const int BUFFER_SIZE = 4000;           // Size of the buffer
volatile int buffer[BUFFER_SIZE];       // Buffer to store analog data
volatile int bufferIndex = 0;           // Current index in the buffer

int val = 0;

void setup() {
  
  
  Serial.begin(9600);
  
  pinMode(MOTOR_PIN, OUTPUT);    // sets the digital pin 13 as output
  myTimer.begin(readIMU, 2000);   
}

void readIMU() {
  
  val = analogRead(A0);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A1);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A2);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A3);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A4);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A5);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A6);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A7);
  buffer[bufferIndex] = val;
  bufferIndex++;

  val = analogRead(A8);
  buffer[bufferIndex] = val;
  bufferIndex++;

  buffer[bufferIndex] = 0;
  bufferIndex++;

}

void loop() {
  if (Serial.available() > 0 ) {
   
    char command = Serial.read();
    
    // Reads Serial Port, and Starts Motor
    if (command == '1') {
      collect_data = true;
      
      Serial.println("VIB ON");
    }

    // Reads Serial Port, and Turns off Motor
    if (command == '0') {
      collect_data = false;
      
      motor_on = false;
      digitalWrite(MOTOR_PIN, motor_on);

      vib_counter = 0;
      
      Serial.println("VIB OFF");
    }
  
  }  

  noInterrupts();

  if (bufferIndex >= BUFFER_SIZE) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
      Serial.print(buffer[i]);
      Serial.print(' ');
      
      if (i % 10 == 9) {
        Serial.println();
      }
    }
  
    bufferIndex = 0; 

    if (collect_data) {
      motor_on = !motor_on;
      digitalWrite(MOTOR_PIN, motor_on);
      vib_counter++;

      if (vib_counter >= MAX_VIB_COUNT) {
        collect_data = false;
        vib_counter = 0;

        Serial.println("VIB OFF");
      }
    }
  }
  
  interrupts();
}