//Written By Nikodem Bartnik - nikodembartnik.pl
#define STEPPER_PIN_1 8
#define STEPPER_PIN_2 9
#define STEPPER_PIN_3 10
#define STEPPER_PIN_4 11
int step_number = 0;
int num_steps = 70;
void setup() {
pinMode(STEPPER_PIN_1, OUTPUT);
pinMode(STEPPER_PIN_2, OUTPUT);
pinMode(STEPPER_PIN_3, OUTPUT);
pinMode(STEPPER_PIN_4, OUTPUT);

}

void loop() {
  Step(num_steps);
  delay(1000);

}
void Step(int num_steps){
  for(int i=0; i<num_steps; i++){
  digitalWrite(STEPPER_PIN_1, step_number==0);
  digitalWrite(STEPPER_PIN_2, step_number==1);
  digitalWrite(STEPPER_PIN_3, step_number==2);
  digitalWrite(STEPPER_PIN_4, step_number==3);
  
  step_number++;
  if(step_number > 3){
    step_number = 0;
    }
  delay(2);
}
}
